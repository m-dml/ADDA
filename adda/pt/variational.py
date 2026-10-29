from typing import Callable, Type

import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.probability.distributions import Distribution
from adda.system.state import State
from adda.util.optimization import optimize
from adda.util.state_space import rollout
from adda.variational.base import process_4dvar_inputs
from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class SC4DVarLoss(_Loss):
    """Loss function class for strong-constraint 4D-Var."""

    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State = None,
        static_inputs: State = None,
        background_prior: Distribution = None,
        param_prior: Distribution = None,
        normalize: bool = False,
        state_time_axis: Tensor = None,
        c_proj: Callable = None,
    ):
        """Initialize loss.

        Args:
            m_dyn (Callable): function that takes 4 inputs: x, dt, dynamic_inputs, static_inputs. All inputs are State
            objects except dt, which can be a float or Tensor. In general, this will be a user-supplied function that
            describes the known dynamics of a state space model.
            observations (ObservationSet): set of observations to be assimilated.
            obs_op (ObservationOperator): observation operator used to map between system states and observations.
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance).
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry).
            background_prior (Distribution): prior distribution on initial state. Defaults to None.
            param_prior (Distribution): prior distribution on parameters.
            normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
            state_time_axis (Tensor): time axis of simulated system states
            c_proj (Callable): optional projection from control-space fields to model-space fields.
                It must take and return a TensorDict with batch size (1, 1).
                The background prior is evaluated in control space before this projection.
        """
        super().__init__()
        (
            self.m_dyn,
            self.observations,
            self.obs_op,
            self.dynamic_inputs,
            self.static_inputs,
            self.background_prior,
            self.parameter_prior,
            self.c_proj,
        ) = (
            m_dyn,
            observations,
            obs_op,
            dynamic_inputs,
            static_inputs,
            background_prior,
            param_prior,
            c_proj,
        )
        self.x0 = State(TensorDict(batch_size=(1, 1)), time_axis=obs_op.time_axis[0:1])
        self.parameters = State(TensorDict(batch_size=(1, 1)), time_axis=obs_op.time_axis[0:1])
        self.normalize = normalize
        self.state_time_axis = state_time_axis if state_time_axis is not None else obs_op.time_axis

    def forward(self, fields: TensorDict, parameter_fields: TensorDict) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for initial system state
            parameter_fields (TensorDict): values for the tuned system parameters

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        self.x0.fields = fields
        self.parameters.fields = parameter_fields
        logp = 0 if self.background_prior is None else self.background_prior.log_prob(self.x0.fields).squeeze(1)
        # logp now has only a single singleton batch dimension
        if self.parameter_prior is not None:
            logp = logp + self.parameter_prior.log_prob(self.parameters.fields)

        if self.c_proj is not None:
            projected_fields = self.c_proj(self.x0.fields)
            self.x0.fields = projected_fields

        x_all = rollout(
            self.m_dyn, self.state_time_axis, self.x0, self.dynamic_inputs, self.static_inputs, self.parameters
        )

        logp = logp + self.obs_op.log_prob(x_all, self.observations).reshape(
            1,
        )
        if self.normalize:
            logp /= self.obs_op.time_axis.nelement()
        return -logp.mean(dim=0)  # mean over batch axis


def sc4dvar_single_window_pt(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationSet,
    x_init: State = None,
    param_init: State = None,
    state_time_axis=None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    background_prior: Distribution = None,
    param_prior: Distribution = None,
    normalize: bool = False,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    parameters_optimizer_pars: dict = {},
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    loss_list: list = None,
    loss_component_list: list = None,
    parameter_list: list = None,
    check_inputs: bool = True,
    c_proj: Callable = None,
    diagnostic_function: Callable = None,
) -> State:
    """Strong-constraint 4D-Var for joint state and parameter estimation.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs.
        observations (ObservationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        x_init (State): first guess of initial state for the first window. Uses background prior mean if None.
          Should have a leading time dimension of length 1.
        param_init (State): first guess for the tuned parameters. Uses parameter prior mean if None.
        state_time_axis (float, optional): the time axis for the state predictions during optimization.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance).
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry).
        background_prior (Distribution): prior distribution on initial state.
        param_prior (Distribution): prior distribution on parameters.
        normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        optimizer_class (Optimizer): class of a pytorch optimizer, default is LBFGS.
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        parameters_optimizer_pars (dict): dictionary containing distinct optimizer settings
            for the optimized parameters.
        n_steps (int): the number of optimization steps.
        verbose (bool): if True, print loss at each iteration. Default is False.
        loss_list (list, optional): Has to be empty. If provided, the loss values are inserted in-place.
        loss_component_list (list, optional): Has to be empty.
            If provided, the loss component values are inserted in-place.
        parameter_list (list, optional): Has to be empty. If provided, the parameter values are inserted in-place at
            each optimizer step.
        check_inputs (bool): if False, input checking is skipped.
        c_proj (Callable): optional projection from control-space fields to model-space fields.
            It must take and return a TensorDict with batch size (1, 1).
            The background_prior is evaluated in control space before this projection.
        diagnostic_function (Callable): Function to call after each optimization step. Should take *x_i* as input.

    Returns:
        x0 (State): optimized initial state
    """
    if check_inputs:
        dynamic_inputs, static_inputs, x_init, state_time_axis, _, _, _, param_init = process_4dvar_inputs(
            obs_op,
            m_dyn,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
            state_time_axis=state_time_axis,
            c_proj=c_proj,
            param_init=param_init,
        )
    if param_init is None:
        if param_prior is not None:
            param_init = State(param_prior.mean.clone().requires_grad_(True), time_axis=state_time_axis[:1])
        else:
            raise ValueError(
                "Either param_init or param_prior must be defined to have an initial guess of the parameters."
            )

    loss = SC4DVarLoss(
        m_dyn,
        observations,
        obs_op,
        dynamic_inputs,
        static_inputs,
        background_prior=background_prior,
        param_prior=param_prior,
        normalize=normalize,
        state_time_axis=state_time_axis,
        c_proj=c_proj,
    )  # instantiate loss

    state_fields, param_fields = optimize(
        loss,
        x_init.fields,
        optimizer_class,
        optimizer_pars,
        scheduler_class,
        scheduler_pars,
        n_steps=n_steps,
        parameters=param_init.fields,
        parameters_optimizer_pars=parameters_optimizer_pars,
        verbose=verbose,
        loss_list=loss_list,
        loss_component_list=loss_component_list,
        parameter_list=parameter_list,
        diagnostic_function=diagnostic_function,
    )
    return State(state_fields, time_axis=x_init.time_axis), State(param_fields, time_axis=param_init.time_axis)


class WC4DVarLoss(_Loss):
    """Base object for representing the weak-constraint 4D-Var loss function."""

    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State,
        static_inputs: State,
        model_error_distribs: Distribution,
        alpha: float = 1.0,
        beta: float = 1.0,
        n_rollout_steps: int = 1,
        background_prior: Distribution = None,
        param_prior: Distribution = None,
        normalize: bool = False,
        state_time_axis: Tensor = None,
    ):
        """Loss for weak constraint 4dvar.

        Args:
            m_dyn (Callable): callable that computes advancement of the state by one time step
            observations (ObservationSet): observations for each time step
            obs_op (ObservationOperator): observation operators for each time step
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
            model_error_distribs (Distribution): probability distributions of the model errors at each time step
            alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
            beta (float, optional): weighting factor for parameter prior loss term. Defaults to 1.0.
            n_rollout_steps (int, optional): number of steps to rollout for the model error term. Defaults to 1.
            background_prior (Distribution, optional): prior on initial state. Defaults to None.
            param_prior (Distribution): prior distribution on parameters.
            normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
            state_time_axis (Tensor, optional): The time axis for the optimized x. Has to be regular.
                If unspecified, tries using obs_op.time_axis instead.
        """
        super().__init__()
        (
            self.m_dyn,
            self.observations,
            self.obs_op,
            self.dynamic_inputs,
            self.static_inputs,
            self.model_error_distribs,
            self.alpha,
            self.beta,
            self.n_rollout_steps,
            self.background_prior,
            self.parameter_prior,
            self.normalize,
        ) = (
            m_dyn,
            observations,
            obs_op,
            dynamic_inputs,
            static_inputs,
            model_error_distribs,
            alpha,
            beta,
            n_rollout_steps,
            background_prior,
            param_prior,
            normalize,
        )
        self.state_time_axis = obs_op.time_axis if state_time_axis is None else state_time_axis
        assert len(self.state_time_axis) >= 2, f"wc4dvar cannot be used with the state time axis {self.state_time_axis}"
        assert torch.allclose(
            self.state_time_axis.diff(), self.state_time_axis.diff()[0], atol=1e-6, rtol=1e-3
        ), f"Only regular state_time_axis are supported for wc4dvar, but the provided one is {self.state_time_axis}"
        self.state_time_axis = state_time_axis
        self.dt = self.state_time_axis.diff()[0]
        self.x = State(TensorDict(batch_size=(1, self.state_time_axis.nelement())), self.state_time_axis)
        self.parameters = State(TensorDict(batch_size=(1, self.state_time_axis.nelement())), self.state_time_axis)

    def forward(self, fields: TensorDict, parameter_fields: TensorDict, component_list=None) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for system state trajectory
            parameter_fields (TensorDict): values for the tuned system parameters
            component_list (list, optional): Has to be empty. If provided, the loss component values
                are inserted in-place

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        if self.n_rollout_steps == 1:
            x = self.x
            x.fields = fields
            x0, x1 = x[:, :-1], x[:, 1:]
        else:
            self.x.fields = self.x.fields.to(fields.device)
            x = self.x.clone()
            x.fields[:, :: self.n_rollout_steps] = fields
            x0, x1 = x[:, :: self.n_rollout_steps][:, :-1], x[:, :: self.n_rollout_steps][:, 1:]
        self.parameters.fields = parameter_fields

        if self.n_rollout_steps == 1:
            x1_pred = self.m_dyn(
                x0,
                self.dt,
                dynamic_inputs=self.dynamic_inputs,
                static_inputs=self.static_inputs,
                parameters=self.parameters,
            )
        else:
            for k in range(self.n_rollout_steps - 1):
                x[:, (k + 1) :: self.n_rollout_steps] = self.m_dyn(
                    x[:, k : -1 : self.n_rollout_steps],
                    self.dt,
                    dynamic_inputs=self.dynamic_inputs[:, k : -1 : self.n_rollout_steps],
                    static_inputs=self.static_inputs,
                    parameters=self.parameters,
                )
            x1_pred = self.m_dyn(
                x[:, self.n_rollout_steps - 1 : -1 : self.n_rollout_steps],
                self.dt,
                dynamic_inputs=self.dynamic_inputs[:, k : -1 : self.n_rollout_steps],
                static_inputs=self.static_inputs,
                parameters=self.parameters,
            )

        logp = 0.0 if self.background_prior is None else self.background_prior.log_prob(x.fields[:, :1]).squeeze(1)

        if component_list is not None:
            assert isinstance(component_list, list), "component_list must be a list"
            _l = []
            if self.background_prior is not None:
                _l.append(-logp.mean(dim=0).item())
        # logp now has only a single singleton batch dimension
        if self.parameter_prior is not None:
            logp = logp + self.beta * self.parameter_prior.log_prob(self.parameters.fields)
            if component_list is not None:
                _l.append(-logp.mean(dim=0).item())
        logp = logp + self.obs_op.log_prob(x, self.observations).reshape(
            1,
        )
        if component_list is not None:
            _l.append(-logp.mean(dim=0).item())
        logp += self.alpha * self.model_error_distribs.log_prob(x1_pred.fields - x1.fields).sum(
            axis=1
        )  # sum over time axis
        if component_list is not None:
            _l.append(-logp.mean(dim=0).item())
            component_list.append([_l[0]] + [b - a for a, b in zip(_l, _l[1:])])
        if self.normalize:
            logp /= self.obs_op.time_axis.nelement()
        return -logp.mean(dim=0)  # mean over batch axis


def wc4dvar_single_window_pt(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    model_error_distribs: Distribution,
    x_init: State = None,
    param_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    n_rollout_steps: int = 1,
    background_prior: Distribution = None,
    param_prior: Distribution = None,
    normalize: bool = False,
    state_time_axis: Tensor = None,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    parameters_optimizer_pars: dict = {},
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    loss_list: list = None,
    loss_component_list: list = None,
    parameter_list: list = None,
    check_inputs: bool = True,
    rollout_init: bool = True,
    diagnostic_function: Callable = None,
):
    """Weak-constraint 4D-Var for a single window of observations. See e.g. equation (28) from [1] and equation (9) from
    [2] for reference. When n_rollout_steps differs from 1, then the scheme is a hybrid between hard- and weak-
    constraint 4D-Var, following [3].

    [1] Carrassi, Alberto, et al. "Data assimilation in the geosciences: An overview
        of methods, issues, and perspectives." Wiley Interdisciplinary Reviews: Climate Change 9.5 (2018): e535.
    [2] Frion, Anthony, et al. "ADDA: a Modular Framework for Representing,
        Simulating and Assimilating Dynamics with End-to-end Differentiability."
        arXiv preprint arXiv:2608.23297 (2026).
    [3] Littaye, Jean, Laurent Memery, and Ronan Fablet.
        "Solving calibration and reanalysis challenges of ocean biogeochemical dynamics
        with neural schemes: a 1D vertical model case-study." EGUsphere 2026 (2026): 1-42.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs.
        observations (ObservationSet): The set of observations on which to compute the observation error.
        obs_op (ObservationOperator): The observation operator used to compute the observation error.
        model_error_distribs: distributions of model errors at each time step
        x_init (State): first guess of state trajectory. Uses background prior mean if None. Should have a leading batch
          dimension, followed by a time dimension.
        param_init (State): first guess for the tuned parameters. Uses parameter prior mean if None.
        state_time_axis (Tensor, optional): the time axis for the optimized x. Has to be regular.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance).
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry).
        alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
        n_rollout_steps (int, optional): number of steps to rollout for the model error term. Defaults to 1.
        background_prior (Distribution): prior distribution of the initial state, defaults to None.
        param_prior (Distribution): prior distribution on parameters.
        normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
            If unspecified, uses the time axis of x_init if possible, otherwise the time axis of obs_op.
        optimizer_class (Optimizer): class of a pytorch optimizer.
        optimizer_pars (dict): dictionary containing parameters for optimizer (learning rate etc.).
        parameters_optimizer_pars (dict): dictionary containing distinct optimizer settings
            for the optimized parameters.
        n_steps (int): indicates the number of optimizer steps.
        verbose (bool): if True, print loss at each iteration. default is False.
        loss_list (list, optional): Has to be empty. If provided, the loss values are inserted in-place.
        loss_component_list (list, optional): Has to be empty.
            If provided, the loss component values are inserted in-place.
        parameter_list (list, optional): Has to be empty. If provided, the parameter values are inserted in-place at
            each optimizer step.
        check_inputs (bool): if False, input checking is skipped.
        rollout_init (bool): indicates whether to copy or rollout the input state if it is only one time step.
        diagnostic_function (Callable): Function to call after each optimization step. Should take *x_i* as input.
    Returns:
        x0 (State): optimized state trajectory. first dimension is batch, second is time.
    """
    if check_inputs:
        (
            dynamic_inputs,
            static_inputs,
            x_init,
            state_time_axis,
            _,
            _,
            model_error_distribs,
            param_init,
        ) = process_4dvar_inputs(
            obs_op,
            m_dyn,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
            model_error_distribs=model_error_distribs,
            n_rollout_steps=n_rollout_steps,
            state_time_axis=state_time_axis,
            rollout_init=rollout_init,
            param_init=param_init,
        )
    if param_init is None:
        if param_prior is not None:
            param_init = State(param_prior.mean.clone().requires_grad_(True), time_axis=state_time_axis[:1])
        else:
            raise ValueError(
                "Either param_init or param_prior must be defined to have an initial guess of the parameters."
            )

    loss = WC4DVarLoss(
        m_dyn,
        observations,
        obs_op,
        dynamic_inputs,
        static_inputs,
        model_error_distribs,
        alpha=alpha,
        beta=beta,
        n_rollout_steps=n_rollout_steps,
        background_prior=background_prior,
        param_prior=param_prior,
        normalize=normalize,
        state_time_axis=state_time_axis,
    )
    fields, param_fields = optimize(
        loss,
        x_init.fields,
        optimizer_class,
        optimizer_pars,
        scheduler_class,
        scheduler_pars,
        n_steps=n_steps,
        parameters=param_init.fields,
        parameters_optimizer_pars=parameters_optimizer_pars,
        verbose=verbose,
        loss_list=loss_list,
        loss_component_list=loss_component_list,
        parameter_list=parameter_list,
        diagnostic_function=diagnostic_function,
    )
    if n_rollout_steps == 1:
        return State(fields, time_axis=state_time_axis), State(param_fields, time_axis=param_init.time_axis)
    else:  # rollout from each optimized time step to re-constitute a full State over time
        full_state = loss.x.clone()
        full_state.fields[:, ::n_rollout_steps] = fields
        for k in range(n_rollout_steps - 1):
            full_state[:, (k + 1) :: n_rollout_steps] = m_dyn(
                full_state[:, k:-1:n_rollout_steps],
                loss.dt,
                dynamic_inputs=loss.dynamic_inputs[:, k:-1:n_rollout_steps],
                static_inputs=loss.static_inputs,
                parameters=State(param_fields, time_axis=param_init.time_axis),
            )
    return full_state, State(param_fields, time_axis=param_init.time_axis)

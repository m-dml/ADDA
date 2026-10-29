from typing import Callable, Type, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def optimize(
    f: Callable,
    x: Union[Tensor, TensorDict],
    optimizer_class: Type[Optimizer],
    optimizer_pars: dict,
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    parameters: Union[Tensor, TensorDict] = None,
    parameters_optimizer_pars: dict = {},
    verbose: bool = False,
    loss_list: list = None,
    loss_component_list: list = None,
    parameter_list: list = None,
    diagnostic_function: Callable = None,
) -> Union[Tensor, TensorDict]:
    """

    Args:
        f (Callable): function to be minimized
        x (Union[Tensor, TensorDict]): initial estimate of inputs
        optimizer_class: class of pytorch optimizer
        optimizer_pars (dict): dict of optimizer parameters
        scheduler_class: class of pytorch learning rate scheduler
        scheduler_pars (dict): dict of scheduler parameters
        n_steps (int): number of optimizer steps
        parameters (Union[Tensor, TensorDict]): initial estimate of tuned parameters
        parameter_optimizer_pars (Dict): optimizer parameters related to the tuned parameters
        verbose (bool): if True, print loss at each iteration. default is False
        loss_list (list, optional): Has to be empty. If provided, the loss values are inserted in-place
        loss_component_list (list, optional): Has to be empty. If provided, the loss component values
            are inserted in-place
        parameter_list (list, optional): Has to be empty. If provided, the parameter values are inserted
            in-place at each optimizer step.
        diagnostic_function (Callable): Function to call after each optimization step. Should take *x_i* as input.

    Returns:
        x (Union[Tensor, TensorDict]): optimized inputs to f
        parameters (Union[Tensor, TensorDict]): optimized parameters of f
    """
    if isinstance(x, TensorDict):
        inputs = [v for v in x.values() if isinstance(v, Tensor)]
    else:
        assert isinstance(x, torch.Tensor), f"x must be a Tensor or TensorDict, got {type(x)}"
        inputs = [x]
    if isinstance(parameters, TensorDict):
        inputs_list = inputs + [p for p in parameters.values() if isinstance(p, Tensor)]
        inputs = [
            {"params": inputs},
            {"params": [p for p in parameters.values() if isinstance(p, Tensor)], **parameters_optimizer_pars},
        ]
    elif isinstance(parameters, torch.Tensor):
        inputs_list = inputs + [parameters]
        inputs = [{"params": inputs}, {"params": parameters, **parameters_optimizer_pars}]
    else:
        assert parameters is None, f"parameters must be a Tensor or TensorDict, got {type(parameters)}"
        inputs_list = inputs

    optimizer = optimizer_class(params=inputs_list if parameters_optimizer_pars == {} else inputs, **optimizer_pars)

    if scheduler_class is not None:
        scheduler_pars = scheduler_pars or {}
        scheduler = scheduler_class(optimizer=optimizer, **scheduler_pars)

    if loss_list is not None:
        assert isinstance(loss_list, list), "loss_list should be None or an empty list"
        assert not loss_list, "loss_list should be None or an empty list, but it is not empty"

    if loss_component_list is not None:
        assert isinstance(loss_component_list, list), "loss_component_list should be None or an empty list"
        assert not loss_component_list, "loss_component_list should be None or an empty list, but it is not empty"

    if parameter_list is not None:
        assert isinstance(parameter_list, list), "parameter_list should be None or an empty list"
        assert not parameter_list, "parameter_list should be None or an empty list, but it is not empty"

    for i_opt in range(n_steps):

        def closure():
            """Closure function that updates and zeros gradients as needed. used by LBFGS which sometimes, but not
            always needs gradients.

            Returns: None
            """
            if parameters is None:
                if loss_component_list is None:
                    loss = f(x)
                else:
                    loss = f(x, component_list=loss_component_list)
            else:
                if loss_component_list is None:
                    loss = f(x, parameters)
                else:
                    loss = f(x, parameters, component_list=loss_component_list)
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
            if verbose:
                print(f"iteration {i_opt}: loss = {loss.item()}")
            if isinstance(loss_list, list):
                loss_list.append(loss.item())
            if parameters is not None and isinstance(parameter_list, list):
                parameter_list.append(parameters.detach().clone().cpu())
            return loss

        optimizer.step(closure)
        if scheduler_class is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                if verbose:
                    print(f"iteration {i_opt}: lr = {param_group['lr']}")

        for param in inputs_list:
            if torch.isnan(param).any():
                raise ValueError("NaN encountered during optimization")

        if diagnostic_function is not None:
            diagnostic_function(x.detach())

    if parameters is None:
        return x
    else:
        return x, parameters

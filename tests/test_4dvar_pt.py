from unittest import TestCase

import torch
from adda.observation.operators import random_sparse_noisy_obs
from adda.probability.distributions import DiagonalGaussian
from adda.pt.variational import sc4dvar_single_window_pt, wc4dvar_single_window_pt
from adda.system.state import State
from adda.util.initialization import naive_initialization
from mdml_sim.lorenz96 import L96Simulator
from tensordict import TensorDict


def setup_task(
    n_variables=40,
    time_step=0.01,
    nb_steps=800,
    nb_steps_burnt=400,
    noise_amplitude=1.0,
    p_obs=0.25,
    forcing=8.0,
):
    true_forcing = forcing + torch.randn(1).item()  # Sample F
    groundtruth = generate_groundtruth(
        n_variables,
        time_step,
        nb_steps,
        nb_steps_burnt=nb_steps_burnt,
        forward_operator=L96Simulator(forcing=true_forcing),
    )
    groundtruth, obs_op, observations = random_sparse_noisy_obs(groundtruth, noise_amplitude, p_obs)

    initialization = naive_initialization(observations)
    parameter_init = State(torch.Tensor([8.0]).reshape(1, 1, 1)).clone().requires_grad_()

    return groundtruth, obs_op, observations, initialization, parameter_init, true_forcing


def generate_groundtruth(
    n_variables, time_step, nb_steps, nb_steps_burnt=400, forward_operator=L96Simulator(forcing=8)
):
    initial_state = torch.randn(1, n_variables)
    nb_steps_total = nb_steps + nb_steps_burnt
    forecast_steps = torch.arange(0, nb_steps_total) * torch.tensor(time_step, dtype=torch.float64)
    time_series = forward_operator.integrate(time=forecast_steps, state=initial_state).squeeze()
    time_series = time_series[nb_steps_burnt:]
    return State(time_series.reshape(1, *time_series.shape), time_axis=forecast_steps[:nb_steps])


def next_step_function(x: torch.Tensor, dt: float, dynamic_inputs: State, static_inputs: State, parameters: State):
    B, T = x.fields.batch_size[:2]
    x_tensor = x.fields.reshape(-1)["x"]  # combine batch and time dimensions, leave others intact
    forward_operator = L96Simulator(
        forcing=parameters.fields["x"].squeeze().expand(x_tensor.shape).squeeze(1), use_adjoint=False
    )
    integrated = forward_operator.integrate(torch.arange(2) * dt, x_tensor)[
        :, 1
    ]  # integrate over a time step, take the current time
    integrated = integrated.unsqueeze(0)  # re-add batch dimension
    new_fields = TensorDict(x=integrated, batch_size=(1, T))
    return State(new_fields)  # return State object, advancing time


class Test4DVar(TestCase):
    def test_sc4dvar_pt(self):
        (
            groundtruth,
            obs_op,
            observations,
            initialization,
            parameter_init,
            true_forcing,
        ) = setup_task()

        n_obs_steps = (
            50  # we assimilate only on the earliest time steps, otherwise the optimization would be too unstable
        )
        time_window = (observations.state.time_axis[0], observations.state.time_axis[n_obs_steps])
        obs_op_opt = obs_op.restrict_time_domain(*time_window)
        observations_opt = observations.restrict_time_domain(*time_window)
        param_prior = DiagonalGaussian(torch.Tensor([8.0]), torch.Tensor([1.0]))

        assimilated_ic, assimilated_params = sc4dvar_single_window_pt(
            next_step_function,
            observations_opt,
            obs_op_opt,
            initialization[:, :1],
            parameter_init,
            param_prior=param_prior,
            optimizer_pars={"lr": 1e-1},
        )
        assimilated_forcing = assimilated_params.fields["x"].item()
        print(f"estimated forcing: {assimilated_forcing}, GT {true_forcing}")
        self.assertTrue(
            abs(true_forcing - assimilated_forcing) < 0.8,
            msg=f"The performance is worse than usual observed results: GT {true_forcing}, estimated {assimilated_forcing}",
        )

    def test_wc4dvar_pt(self):
        (
            groundtruth,
            obs_op,
            observations,
            initialization,
            parameter_init,
            true_forcing,
        ) = setup_task()

        T = groundtruth.time_axis.nelement()
        n_rollout_steps = 10
        # We have to adapt the shape of the model error distribution's parameters according to the n_rollout_steps parameters
        model_error_distribs = DiagonalGaussian(
            torch.zeros(1, (T - 1) // n_rollout_steps, 40), torch.ones(1, (T - 1) // n_rollout_steps, 40)
        )
        param_prior = DiagonalGaussian(torch.Tensor([8.0]), torch.Tensor([1.0]))
        assimilated_states, assimilated_params = wc4dvar_single_window_pt(
            next_step_function,
            observations,
            obs_op,
            model_error_distribs,
            x_init=initialization,
            param_init=parameter_init,
            param_prior=param_prior,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_rollout_steps=n_rollout_steps,
            n_steps=20,
            verbose=0,
        )
        assimilated_forcing = assimilated_params.fields["x"].item()
        self.assertTrue(
            abs(true_forcing - assimilated_forcing) < 0.2,
            msg=f"The performance is worse than usual observed results: GT {true_forcing}, estimated {assimilated_forcing}",
        )

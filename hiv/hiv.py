import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint, ode

"""HIV Treatment domain based on https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains/HIVTreatment.py
Deborah Hanus of Harvard DTAK contributed to the implementation.
"""

# Original attribution information:
__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class HIVTreatment(gym.Env):
    metadata = {}
    eps_values_for_actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])
    state_names = ("T1", "T1*", "T2", "T2*", "V", "E")
    reward_range = (-1e300, 1e300)
    num_actions = 4

    def __init__(self, logspace=True):
        self.discount_factor = 0.98
        self.horizon = 200
        self.model_derivatives = dsdt
        self.dt = 5
        self.logspace = logspace
        self._max_episode_steps = self.horizon

        if logspace:
            high = np.array([8.]*6)
            low = np.array([-5.]*6)
        else:
            high = np.array([1e8] * 6)
            low = np.array([0.] * 6)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = None
        self.start_state = np.array([163573., 5., 11945., 46., 63919., 24.])
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        self.t += 1
        eps1, eps2 = self.eps_values_for_actions[action]
        r = ode(self.model_derivatives).set_integrator('vode', nsteps=10000, method='bdf')
        t0 = 0
        deriv_args = (eps1, eps2)
        r.set_initial_value(self.state, t0).set_f_params(deriv_args)
        self.state = r.integrate(self.dt)
        reward = self.calc_reward(action=action)
        return self.observe(), reward, self.is_done(), {}

    def calc_reward(self, action=0):
        """Calculate the reward for the specified transition."""
        eps1, eps2 = self.eps_values_for_actions[action]
        T1, T2, T1s, T2s, V, E = self.state
        # the reward function penalizes treatment because of side-effects
        reward = -0.1*V - 2e4*eps1**2 - 2e3*eps2**2 + 1e3*E
        # Constrain reward to be within specified range
        reward = np.clip(reward, self.reward_range[0], self.reward_range[1])
        if np.isnan(reward):
            reward = -self.reward_bound
        reward = reward/10**8
        return reward

    def reset(self):
        self.t = 0
        self.state = self.start_state.copy()
        return self.observe()

    def observe(self):
        """Return current state."""
        if self.logspace:
            return np.log10(self.state)
        else:
            return self.state

    def is_done(self):
        """Check if we've finished the episode."""
        return True if self.t >= self.horizon else False

    def render(self, mode='human'):
        return

    def close(self):
        pass
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None

def dsdt(t, s, params):
        """Wrapper for system derivative with respect to time"""
        derivs = np.empty_like(s)
        eps1, eps2 = params
        dsdt_(derivs, s, t, eps1, eps2)
        return derivs

def dsdt_(out, s, t, eps1, eps2):
        """System derivate with respect to time (days).
        Arguments:
        out -- output
        s -- state
        t -- time
        eps1 -- action effect
        eps2 -- action effect
        """
        # baseline model parameter constants
        lambda1 = 1e4  # Target cell, type 1, production rate *CAN BE VARIED*
        lambda2 = 31.98  # Target cell, type 2, production rate *CAN BE VARIED*
        d1 = 0.01  # Target cell, type 1, death rate
        d2 = 0.01  # Target cell, type 2, death rate
        f = .34  # Treatment efficacy, reduction in population 2 \in[0,1] *CAN BE VARIED*
        k1 = 8e-7  # Population 1, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
        k2 = 1e-4  # Population 2, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
        delta = .7  # Infected cell death rate
        m1 = 1e-5  # Immune-induced clearance rate, population 1 *CAN BE VARIED*
        m2 = 1e-5  # Immune-induced clearance rate, population 2 *CAN BE VARIED*
        NT = 100.  # Virions produced per infected cell
        c = 13.  # Virius natural death rate
        rho1 = 1.  # Average number of virions infecting type 1 cell
        rho2 = 1.  # Average number of virions infecting type 2 cell
        lambdaE = 1.  # Immune effector production rate *CAN BE VARIED*
        bE = 0.3  # Maximum birth rate for immune effectors *SENSITVE TO GROWTH, CAN BE VARIED*
        Kb = 100.  # Saturation constant for immune effector birth *CAN BE VARIED*
        d_E = 0.25  # Maximum death rate for immune effectors *CAN BE VARIED*
        Kd = 500.  # Saturation constant for immune effectors death *CAN BE VARIED*
        deltaE = 0.1  # Natural death rate for immune effectors

        # decompose state
        T1, T2, T1s, T2s, V, E = s

        # compute derivatives
        tmp1 = (1. - eps1) * k1 * V * T1
        tmp2 = (1. - f * eps1) * k2 * V * T2
        out[0] = lambda1 - d1 * T1 - tmp1
        out[1] = lambda2 - d2 * T2 - tmp2
        out[2] = tmp1 - delta * T1s - m1 * E * T1s
        out[3] = tmp2 - delta * T2s - m2 * E * T2s
        out[4] = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
                 - ((1. - eps1) * rho1 * k1 * T1 +
                    (1. - f * eps1) * rho2 * k2 * T2) * V
        out[5] = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
                 - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

try:
    import numba
except ImportError as e:
    print("Numba acceleration unavailable, expect slow runtime.")
else:
    dsdt_ = numba.jit(
        numba.void(numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64),
        nopython=True, nogil=True)(dsdt_)
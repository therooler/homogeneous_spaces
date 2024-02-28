import numpy as np


class ADAM(object):
    def __init__(self, params, learning_rate: float, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.learning_rate = learning_rate
        self._momentums = np.zeros_like(params)
        self._velocities = np.zeros_like(params)
        self.iterations = 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_params(self, gradient):
        local_step = self.iterations + 1,
        beta_1_power = np.power(self.beta_1, local_step)
        beta_2_power = np.power(self.beta_2, local_step)
        alpha = self.learning_rate * np.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        self._momentums += (gradient - self._momentums) * (1 - self.beta_1)
        self._velocities += (gradient ** 2 - self._velocities) * (1 - self.beta_2)
        update_step = (self._momentums * alpha) / (np.sqrt(self._velocities) + self.epsilon)
        return update_step

import numpy as np


class Adam:
    """
    Adaptive Moment Estimation (Adam) Optimizer.
    Includes bias correction for the first few steps.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer) -> None:
        if not hasattr(layer, "params") or not layer.params:
            return

        self.t += 1
        layer_id = id(layer)

        if layer_id not in self.m:
            self.m[layer_id] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            self.v[layer_id] = {k: np.zeros_like(v) for k, v in layer.params.items()}

        for key in layer.params.keys():
            grad = layer.grads[key]

            # Momentum
            self.m[layer_id][key] = (
                self.beta1 * self.m[layer_id][key] + (1 - self.beta1) * grad
            )
            # RMSProp
            self.v[layer_id][key] = self.beta2 * self.v[layer_id][key] + (
                1 - self.beta2
            ) * (grad**2)

            # Bias correction
            m_hat = self.m[layer_id][key] / (1 - self.beta1**self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta2**self.t)

            # Update parameters
            layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

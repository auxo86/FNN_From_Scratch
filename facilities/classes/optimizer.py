import cupy as cp


class Adam:
    def __init__(self, fLearningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.fLearningRate = fLearningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, parameters, gradients):
        self.t += 1
        for key in parameters:
            if key not in self.m:
                self.m[key] = cp.zeros_like(parameters[key])
                self.v[key] = cp.zeros_like(parameters[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * cp.square(gradients[key])
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            parameters[key] -= self.fLearningRate * m_hat / (cp.sqrt(v_hat) + self.epsilon)
            #if m_hat.shape == (1, 15):
            #    print(f'm_hat[0, 0]: {m_hat[0, 0]}, v_hat[0,0]: {v_hat[0, 0]}')

        return parameters

    # 在每個Epoch開始時重置Adam優化器的狀態
    def ResetAdam(self):
        self.m = {}
        self.v = {}
        self.t = 0


class SGD:
    def __init__(self, fLearningRate=0.001):
        self.fLearningRate = fLearningRate

    def update(self, parameters, gradients):
        for key in parameters:
            parameters[key] -= self.fLearningRate * gradients[key]

        return parameters

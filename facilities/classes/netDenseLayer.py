import numpy as np

from facilities.StaticFuncs import he_initialization


# 要有輸入的維度和輸出的維度
# 還有指定活化函數
# Dense layer 的初始化最重要的是初始化 w, b, 活化函數
class Dense:
    def __init__(self, iInputNeuronNum, iOutputNeuronNum, ActivationFunc=None):
        self.iInNeuronNum = iInputNeuronNum
        self.iNeuronNum = iOutputNeuronNum
        # 初始化權重( 平均值 0 標準差 1 )，用輸入輸出的維度來決定權重的個數。
        # np.random.randn() 傳回平均值 0 標準差 1 的浮點數序列
        # 初始化權重，使用 He initialization
        # Xavier initialization 主要適用於使用 tanh、sigmoid 等 S 型函數的情況，
        # 而 He initialization 主要適用於使用 ReLU、LeakyReLU 等 ReLU 相關函數的情況。
        # 這是因為，將權重初始化為較小的隨機值，可以使其輸出值較小，從而更容易進行調整和更新。
        self.W = he_initialization(self.iInNeuronNum, self.iNeuronNum)
        # bias 只需要跟輸出維度一樣多就好
        self.b = np.zeros((1, iOutputNeuronNum))
        self.naInput = None
        self.dW = None
        self.db = None
        # 挑選想要的活化函數。
        self.activation = ActivationFunc

    # 往前傳播，X 是輸入層
    def forward(self, naIn, boolIfTraining=True):
        self.naInput = naIn
        self.naOutput = np.dot(self.naInput, self.W) + self.b
        if self.activation is not None:
            return self.activation(self.naOutput)
        else:
            return self.naOutput

    # 往後傳播， naOutDiff 是上一層的 backward 的輸出。
    def backward(self, naOutDiff):
        if self.activation is not None:
            # 在一般隱藏層的 backward 中，我們需要先計算出這一層的激活函數的導數。
            # naDelta = np.dot(naOutDiff, self.W.T) * self.activation(self.naOutput, derivative=True)
            naDelta = naOutDiff * self.activation(self.naOutput, derivative=True)
        else:
            naDelta = naOutDiff

        m = self.naInput.shape[0]
        self.dW = 1 / m * np.dot(self.naInput.T, naDelta)
        self.db = 1 / m * np.sum(naDelta, axis=0, keepdims=True)

        return np.dot(naDelta, self.W.T)

    def update(self, fLearningRate):
        self.W -= fLearningRate * self.dW
        self.b -= fLearningRate * self.db
        # # 初始化
        # m_W, m_b = np.zeros_like(self.W), np.zeros_like(self.b)
        # v_W, v_b = np.zeros_like(self.W), np.zeros_like(self.b)
        # beta1, beta2 = 0.9, 0.999
        # epsilon = 1e-8
        # t = 0
        #
        # # 計算梯度
        # # dW, db = self.backward(dA)
        #
        # # 更新m, v
        # m_W = beta1 * m_W + (1 - beta1) * self.dW
        # v_W = beta2 * v_W + (1 - beta2) * (self.dW ** 2)
        # m_b = beta1 * m_b + (1 - beta1) * self.db
        # v_b = beta2 * v_b + (1 - beta2) * (self.db ** 2)
        #
        # # 更新W, b
        # t += 1
        # m_W_corr = m_W / (1 - beta1 ** t)
        # v_W_corr = v_W / (1 - beta2 ** t)
        # m_b_corr = m_b / (1 - beta1 ** t)
        # v_b_corr = v_b / (1 - beta2 ** t)
        # self.W -= fLearningRate * m_W_corr / (np.sqrt(v_W_corr) + epsilon)
        # self.b -= fLearningRate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)
import cupy as cp

from facilities.StaticFuncs import he_initialization
from facilities.classes.optimizer import Adam


# 要有輸入的維度和輸出的維度
# 還有指定活化函數
# Dense layer 的初始化最重要的是初始化 w, b, 活化函數
class Dense:
    def __init__(self, iInputNeuronNum, iOutputNeuronNum, ActivationFunc=None, Optimizer=Adam):
        self.optimizer = Optimizer()
        self.iInNeuronNum = iInputNeuronNum
        self.iNeuronNum = iOutputNeuronNum
        # 初始化權重( 平均值 0 標準差 1 )，用輸入輸出的維度來決定權重的個數。
        # cp.random.randn() 傳回平均值 0 標準差 1 的浮點數序列
        # 初始化權重，使用 He initialization
        # Xavier initialization 主要適用於使用 tanh、sigmoid 等 S 型函數的情況，
        # 而 He initialization 主要適用於使用 ReLU、LeakyReLU 等 ReLU 相關函數的情況。
        # 這是因為，將權重初始化為較小的隨機值，可以使其輸出值較小，從而更容易進行調整和更新。
        self.W = he_initialization(self.iInNeuronNum, self.iNeuronNum)
        # bias 只需要跟輸出維度一樣多就好
        self.b = cp.zeros((1, iOutputNeuronNum))
        self.naInput = None
        self.dW = None
        self.db = None
        # 挑選想要的活化函數。
        self.activation = ActivationFunc

    # 往前傳播，X 是輸入層
    def forward(self, naIn, boolIfTraining=True):
        self.naInput = naIn
        self.naOutput = cp.dot(self.naInput, self.W) + self.b
        if self.activation is not None:
            return self.activation(self.naOutput)
        else:
            return self.naOutput

    # 往後傳播， naOutDiff 是上一層的 backward 的輸出。
    def backward(self, naOutDiff):
        if self.activation is not None:
            # 在一般隱藏層的 backward 中，我們需要先計算出這一層的激活函數的導數。
            # naDelta = cp.dot(naOutDiff, self.W.T) * self.activation(self.naOutput, derivative=True)
            naDelta = self.activation(self.naOutput, derivative=True) * naOutDiff
        else:
            naDelta = naOutDiff

        m = self.naInput.shape[0]
        self.dW = 1 / m * cp.dot(self.naInput.T, naDelta)
        self.db = 1 / m * cp.sum(naDelta, axis=0, keepdims=True)

        return cp.dot(naDelta, self.W.T)

    def update(self, fLearningRate):
        self.optimizer.fLearningRate = fLearningRate
        dictParams = {'W': self.W, 'b': self.b}
        dictGradients = {'W': self.dW, 'b': self.db}

        dictParams = self.optimizer.update(parameters=dictParams, gradients=dictGradients)
        self.W = dictParams['W']
        self.b = dictParams['b']
        #if self.__class__.__name__ == 'MultiClassOutputLayer':
        #    print(f'self.W[0,0] = {self.W[0, 0]}, self.b = {self.b[0, 0]}')
            
    def resetAdam(self):
        self.optimizer.ResetAdam()

import cupy as cp

from facilities.StaticFuncs import softmax, CrossEntropy, he_initialization
from facilities.classes.netDenseLayer import Dense
from facilities.classes.optimizer import Adam


# 這個輸出層在 forward 方法中使用了 softmax 函數將輸出轉換成概率分佈，
# 並且計算了交叉熵損失函數，並將結果存在 self.loss 中。
# 在 backward 方法中，計算了self.loss，並將其返回。
# 這個輸出誤差 self.loss 將被傳遞到前面的層中，以計算每個參數的梯度，進而進行反向傳播更新權重和偏置。

# 在這個多元分類輸出層的實作中，我們使用 softmax 函數來獲得輸出的機率分佈，
# 然後使用交叉熵損失函數來計算輸出與真實標籤之間的差異。
# 在 backward 中，我們計算出損失函數的梯度再傳播。
# 在更新權重和偏差時，我們使用輸出層的差異計算梯度，進行反向傳播。
class MultiClassOutputLayer(Dense):
    def __init__(self, iInputNeuronNum, iOutputNeuronNum, ActivationFunc=softmax, Optimizer=Adam):
        super().__init__(iInputNeuronNum, iOutputNeuronNum)
        self.W = he_initialization(self.iInNeuronNum, self.iNeuronNum)
        # bias 只需要跟輸出維度一樣多就好
        self.b = cp.zeros((1, iOutputNeuronNum))
        self.naOutput = None
        self.naResultP = None
        self.loss = None
        self.activation = ActivationFunc
        self.optimizer = Optimizer()

    # 重寫 forward
    def forward(self, naIn, boolIfTraining=True):
        self.naInput = naIn
        self.naOutput = cp.dot(self.naInput, self.W) + self.b
        self.naResultP = self.activation(self.naOutput)  # 使用 softmax 激活函數
        return self.naResultP

    def backward(self, naTrueLabels):
        naOrigDelta = self.naResultP - naTrueLabels
        delta = self.activation(self.naOutput, derivative=True) * naOrigDelta
        iNumSamples = len(self.naOutput)

        # 更新權重和偏差
        self.dW = cp.dot(self.naInput.T, delta) / iNumSamples
        self.db = cp.sum(delta, axis=0) / iNumSamples
        return cp.dot(delta, self.W.T)


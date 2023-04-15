import numpy as np


class netFNNSequential:
    def __init__(self, iBatchSize, fMaxLearningRate, CalLossFunc):
        self.listTrueLabels = None
        self.listPredLabels = None
        self.fAccuracy = None
        self.naTrueLabels = None
        self.boolIfTraining = None
        self.loss = None
        self.naDiffOut = None
        self.errDiffOut = None
        self.layers = []
        self.fLearningRate = fMaxLearningRate
        self.naOut = None
        self.iBatchSize = iBatchSize
        self.calLossFunc = CalLossFunc

    def add(self, layer):
        self.layers.append(layer)

    # 在 forward 方法中返回神經網路的預測結果。
    def forward(self, naInput, boolIfTraining=True):
        self.boolIfTraining = boolIfTraining
        self.naOut = naInput
        for layer in self.layers:
            self.naOut = layer.forward(self.naOut, self.boolIfTraining)
        # 為了把 self.naOut 餵到 backward 函數
        self.naDiffOut = self.naOut

    def backward(self, naDiffout):
        # 例如你有一個 list 是 [1,2,3,4,5]
        # 套了 reversed 就會倒過來做 iteration 變成 [5,4,3,2,1]
        self.naDiffOut = naDiffout
        for layer in reversed(self.layers):
            self.naDiffOut = layer.backward(self.naDiffOut)

    def update(self):
        for layer in self.layers:
            layer.update(self.fLearningRate)

    def ComputeLoss(self, naLbls):
        # 計算 softmax 產生機率的 cross entropy loss.
        self.loss = self.calLossFunc(self.naOut, naLbls)  # 使用 CrossEntropy

    def training(self, naSampleTrain, naLabelTrain):
        self.forward(naSampleTrain, boolIfTraining=True)
        self.backward(naLabelTrain)
        self.update()

    def validation(self, naTestImages, naTestOneHotLabels):
        self.naTrueLabels = naTestOneHotLabels
        self.forward(naTestImages, boolIfTraining=False)
        self.ComputeLoss(self.naTrueLabels)

    def TransToListLabels(self, naX):
        # 將結果轉換為具體的分類標籤
        return np.argmax(naX, axis=1)
        
    def CalAccuracy(self):
        self.listPredLabels = self.TransToListLabels(self.naOut)
        self.listTrueLabels = self.TransToListLabels(self.naTrueLabels)
        iSumCorrectPredictions = np.sum(self.listPredLabels == self.listTrueLabels)
        self.fAccuracy = iSumCorrectPredictions/len(self.naTrueLabels)

    def CalLearningRate(self, fAccuracy):
        fLearningRate = (1-fAccuracy)/3
        return fLearningRate
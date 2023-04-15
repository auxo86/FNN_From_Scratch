# 定義 DropoutLayer Layer
import numpy as np


class DropoutLayer:
    def __init__(self, fDropoutRate):
        self.dropoutMask = None
        self.fDropoutRatio = fDropoutRate

    def forward(self, naIn, boolIfTraining=True):
        if boolIfTraining:
            # 產生 mask
            # *naTest.shape 將 tuple 展開為 naTest.shape[0], naTest.shape[1], ...
            # 回傳的 mask 是 bool 構成的 ndarray
            self.dropoutMask = np.random.rand(*naIn.shape) > self.fDropoutRatio
            # 為了符合期望值所以這裡要 / (1.0 - self.fDropoutRatio)
            return naIn * self.dropoutMask / (1.0 - self.fDropoutRatio)
        else:
            return naIn

    # 通常情況下，我們會將最後一層隱藏層的輸出經過 softmax 函數轉換為概率分布，
    # 然後計算 cross-entropy loss，最終將 loss 送入 backward 函數進行反向傳播。
    def backward(self, naOutDiff):
        return naOutDiff * self.dropoutMask / (1.0 - self.fDropoutRatio)

    # 要實做 update 函數
    def update(self, fLearningRate):
        pass

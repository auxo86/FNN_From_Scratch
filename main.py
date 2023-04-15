import numpy as np

from facilities.StaticFuncs import GetImgAndOneHotEncodedLabel, funcReshapeAndNomalization, relu, softmax, CrossEntropy, \
    GetChunk, CalLearningRate, iNewH, iNewW
from facilities.classes.DropoutLayer import DropoutLayer
from facilities.classes.netDenseLayer import Dense
from facilities.classes.netFNNSequential import netFNNSequential
from facilities.classes.netOutputLayer import MultiClassOutputLayer

if __name__ == '__main__':
    # 資料集分成 training 和 validation 要使用的比例
    floatTrainingDataSizeRatio = 0.8
    # 一次 training 多少資料。
    iTrainingBatchSize = 32
    # 整個資料集要 training 幾次
    iEpochs = 40
    # 學習率
    fMaxLearningRate = 0.9
    sDatasetDir = './ChineseMNIST'

    # 讀資料
    naTrainImages, naTrainOneHotLabels, naTestImages, naTestOneHotLabels, dictLabelCat = GetImgAndOneHotEncodedLabel(sDatasetDir)
    # 每個樣本 reshape 成一維並根據要求作正規化(/255)
    naTrainImages, naTestImages = tuple(map(lambda x: funcReshapeAndNomalization(x), [naTrainImages, naTestImages]))
    # 標準化資料和 one-hot encoding 標籤準備好了，可以進行 training

    fnnModel = netFNNSequential(iTrainingBatchSize, fMaxLearningRate, CrossEntropy)
    # Input layer 是神經網路的第一層，主要是負責接收外部輸入資料，
    # 因此不需要進行加權和運算。
    # 在 Input layer 中，神經元的數量通常等於輸入資料的特徵數量。
    # 例如，如果輸入資料是一個 28x28 的圖片，那麼 Input layer 的神經元數量就是 784（28 naX 28）。
    fnnModel.add(Dense(iNewH * iNewW, 512, relu))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(512, 256, relu))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(256, 64, relu))
    fnnModel.add(DropoutLayer(0.1))
    fnnModel.add(Dense(64, 32, relu))
    fnnModel.add(MultiClassOutputLayer(32, 15, softmax))
    # --------------------------------------------------------------------
    for iepoch in range(iEpochs):
        # 重排元素的順序
        naTrainDsIndics = np.random.permutation(len(naTrainImages))
        naTrainImages, naTrainOneHotLabels = naTrainImages[naTrainDsIndics], naTrainOneHotLabels[naTrainDsIndics]
        for naLittleBatchTrainSample, naLittleBatchTrainLabel in GetChunk(naTrainImages, naTrainOneHotLabels, fnnModel.iBatchSize):
            fnnModel.training(naLittleBatchTrainSample, naLittleBatchTrainLabel)
            fnnModel.validation(naTestImages, naTestOneHotLabels)
            fnnModel.CalAccuracy()

        # 一個 Epoch 中顯示 loss 有多少
        print(f'Epoch {iepoch}: {fnnModel.loss}, Accuracy: {round(fnnModel.fAccuracy*100, 3)}%, learning rate: {fnnModel.fLearningRate}')
        # 更新 learning rate
        fnnModel.fLearningRate = CalLearningRate(fnnModel.fAccuracy, fMaxLearningRate)

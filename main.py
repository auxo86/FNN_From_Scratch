import cupy as cp

from facilities.StaticFuncs import GetImgAndOneHotEncodedLabel, funcReshapeAndNomalization, relu, softmax, CrossEntropy, \
    GetChunk, CalLearningRate, iNewH, iNewW, leaky_relu
from facilities.classes.DropoutLayer import DropoutLayer
from facilities.classes.netDenseLayer import Dense
from facilities.classes.netFNNSequential import netFNNSequential
from facilities.classes.netOutputLayer import MultiClassOutputLayer
from facilities.classes.optimizer import SGD, Adam
from facilities.classes.AccuracyPlotter import AccuracyPlotter

if __name__ == '__main__':
    # 資料集分成 training 和 validation 要使用的比例
    floatTrainingDataSizeRatio = 0.8
    # 一次 training 多少資料。
    iTrainingBatchSize = 32
    # 整個資料集要 training 幾次
    iEpochs = 70
    
    # 設定 optimizer
    optimizer = SGD
    
    # 學習率
    if optimizer == Adam:
        fMaxLearningRate = 0.0001
    elif optimizer == SGD:
        fMaxLearningRate = 0.9
    
    sDatasetDir = './ChineseMNIST'
    
    # 繪製折線圖
    plotter = AccuracyPlotter('Epochs', 'Accuracy', iEpochs)

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
    fnnModel.add(Dense(iNewH * iNewW, 2048, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.3))
    fnnModel.add(Dense(2048, 1024, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(1024, 512, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(512, 256, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(256, 128, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.2))
    fnnModel.add(Dense(128, 64, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.1))
    fnnModel.add(Dense(64, 32, leaky_relu, optimizer))
    fnnModel.add(DropoutLayer(0.1))
    fnnModel.add(MultiClassOutputLayer(32, 15, softmax, optimizer))
    # --------------------------------------------------------------------
    for iepoch in range(iEpochs):
        # 重排元素的順序
        naTrainDsIndics = cp.random.permutation(len(naTrainImages))
        naTrainImages, naTrainOneHotLabels = naTrainImages[naTrainDsIndics], naTrainOneHotLabels[naTrainDsIndics]
        for naLittleBatchTrainSample, naLittleBatchTrainLabel in GetChunk(naTrainImages, naTrainOneHotLabels, fnnModel.iBatchSize):
            fnnModel.training(naLittleBatchTrainSample, naLittleBatchTrainLabel)
            fnnModel.fTrainAccuracy = fnnModel.CalAccuracy()
            fnnModel.validation(naTestImages, naTestOneHotLabels)
            fnnModel.fValidateAccuracy = fnnModel.CalAccuracy()
        # 每一個 epoch 都必須 reset Adam 才能保證 Adam optimizer 的穩定性
        if optimizer == Adam:
            fnnModel.resetAdam()

        # 一個 Epoch 中顯示 loss 有多少
        print(f'Epoch {iepoch}: {fnnModel.loss}, TrainAccuracy: {round(fnnModel.fTrainAccuracy*100, 3)}%, ValidateAccuracy: {round(fnnModel.fValidateAccuracy*100, 3)}%, learning rate: {fnnModel.fLearningRate}')
        # 畫出精確度比較折線圖
        plotter.update(fnnModel.fTrainAccuracy, fnnModel.fValidateAccuracy)
        
        # 更新 learning rate
        fnnModel.fLearningRate = CalLearningRate(fnnModel.fValidateAccuracy, fMaxLearningRate)
        # print(f'fnnModel.layers[-1].W[0, 0], .b[0, 0]: {fnnModel.layers[-1].W[0, 0]}, {fnnModel.layers[-1].b[0, 0]}')

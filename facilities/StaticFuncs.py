import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import image

height, width = 64, 64
fResizeRatio = 0.5
iNewH, iNewW = int(fResizeRatio * height), int(fResizeRatio * width)


def GetImgAndOneHotEncodedLabel(data_dir):
    # 處理標籤
    naMetaDataOfCmnist = pd.read_csv(f'{data_dir}/chinese_mnist.csv').to_numpy()
    # 去掉 character(用來製造檔名)
    naOfLabelData = naMetaDataOfCmnist[:, :-1]
    # 只取 character(label)
    naLabels = naMetaDataOfCmnist[:, 4]
    dictLabelCat = {}

    # 取得用於 one-hot encoding 的 dict
    for i, lbl in enumerate(set(naLabels)):
        dictLabelCat[lbl] = i

    # 處理圖像
    sImgDir = os.path.join(data_dir, 'data')
    slistImgFileNames = list(map(lambda x: f'input_{x[0]}_{x[1]}_{x[2]}.jpg', naOfLabelData))
    iNumImgs = len(slistImgFileNames)

    # 真正存手寫影像檔案的 list
    naImages = np.zeros((iNumImgs, iNewH, iNewW))

    # 利用迴圈把所有影像檔案讀進來放到 naImages
    for i, fImgFile in enumerate(slistImgFileNames):
        sImgPath = os.path.join(f'{sImgDir}/', fImgFile)
        rawImg = Image.open(sImgPath).resize((iNewH, iNewW), Image.ANTIALIAS)
        rawImg.save(f'{sImgDir}/resized/{fImgFile}')
        sResizedImgPath = f'{sImgDir}/resized/{fImgFile}'
        # 利用 matplotlib 根據路徑讀入影像檔案塞到 element 中
        img = image.imread(sResizedImgPath)
        naImages[i] = img

    # 將數據集拆分為訓練集和測試集
    iTrainNum = int(iNumImgs * 0.8)
    naIndices = np.random.permutation(iNumImgs)
    naTrainIndices, naTestIndices = naIndices[:iTrainNum], naIndices[iTrainNum:]

    naTrainImages, naTrainLabels = naImages[naTrainIndices], naLabels[naTrainIndices]
    naTestImages, naTestLabels = naImages[naTestIndices], naLabels[naTestIndices]

    # 幫 label 做 one-hot encoding
    naTrainOneHotLabels, naTestOneHotLabels = tuple(
        map(lambda x: OnehotEncode(x, dictLabelCat), [naTrainLabels, naTestLabels])
    )

    return naTrainImages, naTrainOneHotLabels, naTestImages, naTestOneHotLabels, dictLabelCat


def funcReshapeAndNomalization(naImgs: np.ndarray):
    naImgs = naImgs.reshape(naImgs.shape[0], -1).astype('float32') / 255
    return naImgs


def OnehotEncode(naLabels, dictLabelCat):
    naOneHotLabels = np.zeros((len(naLabels), len(dictLabelCat)))
    for i, lbl in enumerate(naLabels):
        naOneHotLabels[i, dictLabelCat[lbl]] = 1
    return naOneHotLabels


# ReLU 激活函數
def relu(naX, derivative=False):
    if derivative:
        return np.where(naX > 0, 1, 0)
    return np.maximum(naX, 0)

# softmax 激活函数
# softmax 函數的作用是將輸入向量轉換為一個概率分布，
# 其中每個元素都是非負數且總和為 1。
# 這樣做的好處是，將模型輸出轉換為概率之後，就可以使用交叉熵等損失函數進行訓練和評估，而交叉熵等損失函數通常要求輸入為概率。
# 另外因為softmax運算的數值範圍通常會很大，可能會超過計算機所能表示的範圍，
# 因此會產生數值上的不穩定，進而影響計算的準確性。
# 所以我們需要對每個元素減去輸入向量中的最大值，使得所有的元素都小於等於0
def softmax(x, derivative=False):
    if derivative:
        # 計算 softmax 函數的導數
        # x 是輸出層輸出的未經 softmax 處理的結果，形狀為 (batch_size, num_classes)
        p = softmax(x)
        # 將每個樣本的概率向量轉換為一個行向量。
        p = p.reshape(p.shape[0], -1)
        return p * (1 - p)
    else:
        # 计算 softmax 函数
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)



# CrossEntropy 損失函數:
# 由於 log 函數在 0 處的值為負無限大，因此需要加上一個很小的 delta 以避免出現除以 0 的情況。
def CrossEntropy(naPred, naTrue):
    # 計算樣本數量
    m = naTrue.shape[0]

    # 避免出現 log(0) 的情況，加上一個極小值 epsilon
    epsilon = 1e-7

    # 計算 cross entropy loss
    loss = -1 / m * np.sum(naTrue * np.log(naPred + epsilon))

    return loss


# 這個函數的輸入是一個二維數組 naOut，代表網絡的輸入或某一層的輸出；
# dropout_ratio 是 dropout 的比例，代表要隨機丟棄的神經元的比例。
# 在函數中，首先創建一個隨機的二維數組 mask，其元素是從均勻分佈 [0, 1) 中隨機生成的。
# 然後，通過比較每個元素和 dropout_ratio 的大小，來判斷是否將其設置為 0，這樣對應的神經元就被丟棄了。
# 最後，為了保持輸入的期望值不變，我們將被保留下來的神經元的值除以 (1.0 - dropout_ratio)，這樣可以保持輸入數值的期望值不變。
def dropout(x, dropout_ratio=0.5):
    mask = np.random.rand(*x.shape) > dropout_ratio
    return mask * x / (1.0 - dropout_ratio)


import numpy as np


def he_initialization(iNumNeuronsIn, iNumNeuronsOut):
    # He initialization: 權重的 He 初始化
    # 參數說明:
    #     iNumNeuronsIn (int): 層中的輸入神經元數.
    #     iNumNeuronsOut (int): 層中的輸出神經元數.
    # Returns:
    #     np.ndarray: 大小為（iNumNeuronsIn，iNumNeuronsOut）的數組，包含初始化的權重。
    w = np.random.randn(iNumNeuronsIn, iNumNeuronsOut) * np.sqrt(2 / iNumNeuronsIn)
    return w


#  假如 my_list 只有 10 個元素，所以使用 my_list[9:12] 時，結束位置 12 已經超出了序列範圍。
#  根據 Python 的切片操作規則，結束位置會自動調整為序列的末尾，因此這個切片操作相當於 my_list[9:10]，
#  會返回一個只包含元素 10 的列表。
#  簡單的說，最後一個批量不需要擔心超出 list 範圍的問題。
def GetChunk(naSample, naLabel, iChunkSize):
    for i in range(0, len(naSample), iChunkSize):
        yield naSample[i:i + iChunkSize], naLabel[i:i + iChunkSize]

def CalLearningRate(fAccuracy, fMaxLearningRate):
    # fLearningRate = ((1 - fAccuracy) / 5)**0.2
    fLearningRate = 2 - np.exp(fAccuracy)*0.2
    if fAccuracy > 0.9:
        fLearningRate -= 0.1*fAccuracy
    fLearningRate = min(fLearningRate, fMaxLearningRate)
    if fLearningRate <= 0:
        fLearningRate = 0.005
    return fLearningRate
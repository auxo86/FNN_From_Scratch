# Chinese MNIST Dataset - FNN

這個專案是基於 [Chinese MNIST Dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist) 所建立的 FNN 神經網路，使用了 Cupy 和 NumPy 來加速計算，並且手動實作了 SGD 和 Adam optimizer 來訓練模型。該模型的目的是識別中文數字。

## 如何使用

1. 安裝相依套件

```
pip install numpy
pip install cupy-cudaXXX # 根據你的 CUDA 版本選擇
```

2. 請大家下載好了解壓縮然後置於 ChineseMNIST 資料夾中。

3. 執行 `python3 main.py` 來開始訓練模型，執行以下指令：

```
python3 main.py
```

4. 訓練好的模型權重目前不會儲存。

## 模型架構

這個 FNN 模型包含了 隱藏層和輸出層。輸入層有 4096 個神經元，輸出層有 15 個神經元（每一個中文數字對應一個神經元）。

模型的架構如下：

```
INPUT -> hidden layer -> optimizer -> hidden layer  -> optimizer... -> output layer -> SOFTMAX -> Final OUTPUT
```

## 訓練過程

使用了 SGD 和 Adam optimizer 來訓練模型，其中 SGD 學習率為 0.9，Adam 學習率為 0.001，batch size 為 32，共訓練了 70 個 epochs。

訓練期間，使用了 Cross Entropy 損失函數來度量模型的表現。在每個 epoch 結束時，都會計算並顯示當前的 loss 和 accuracy，以及測試集上的 loss 和 accuracy。最終測試集上的 accuracy 達到了 85% 以上。

## 結論

本專案展示了如何使用 Cupy 和 NumPy 加速深度學習模型的訓練過程，同時實作了 SGD 和 Adam optimizer 來訓練 FNN 模型。該模型能夠識別中文數字，並且在測試集上達到了 85% 以上的準確率。

# TA_class03

- TA_class03.ipynb (with MNIST datasets)
    - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matteosoo/AI_course/blob/master/course03/TA_class03.ipynb)
- TA_class03_2.ipynb (with Dogs-and-cats datasets)
    - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matteosoo/AI_course/blob/master/course03/TA_class03_2.ipynb)
    - Datasets link: https://drive.google.com/drive/folders/17tih4BezlICt_yNyT022jrBP7XYeGycj?usp=sharing
    - If you want to use colab to run the program, you can add the folder under your Google drive. Note that to mount the drive before you run the code. 

## Deep Neural Networks, DNN
![](https://i.imgur.com/HccJFE1.png)

### Problems of DNN
- DNN的全連接結構會帶來「參數數量膨脹」的問題
    - ex: 解析度 1080 * 960 的圖，光輸入就有1M多個節點，參數數量就有 10^6 * L1 * L2...個(L為該層節點數)
- 彈性決定誰連誰不連的結構又會不利於做Back propagation
- 雖然可以針對輸入特性做前處理(壓縮、切片......etc)
- 但能不能從其他的層面解決問題，比方說換模型？

## Convolutional Neural Networks, CNN 卷積神經網路
- 簡稱CNN
- 原理很好理解，但實際操作起來有點難度
- 跟DNN比起來，收斂速度快很多，訓練參數也少
- 不會因為圖片變形就增加訓練難度(圖片的表達能力提高了)

### Architecture
主要由3種layers相互堆疊
1. Convlutional layer (卷積層)
2. Pooling layer (池化層)
3. Fully-Connected Layer (全連接)

#### 1. Convlutional layer (卷積層)
包含過濾器(Filter, Kernel, Feature Detector)跟特徵圖(Feature Map)
- 圖片進來都必須先給過濾器掃一輪作卷積以生成特徵圖
- 過濾器的內容通常是透過學習過程調整的
- 越深層的過濾器，其代表的意涵會越抽象
- 卷積核(Kernel):
    - 又稱為Filters、Features Detectors
    - 此圖像素 5 * 5，透過卷積運算後(如下gif演示)，將會變成像素 3 * 3 圖像
    - 其中，黃色3*3的Kernel，其值是預訓練的權重，通常用常態分佈隨機產生，再經由訓練更新。因此不要誤會圖中的x1,x0，其只是為了計算方便，不然應該都是一些隨機或常態分佈的小數。
    - ![](https://i.imgur.com/RSX2cIV.gif)
    - 縮小程度之兩個原因
        - Border effect
        - strides (步長)

#### 2. Pooling layer (池化層)
- 別名Spatial Pooling, Subsampling, Downsampling
- 概念是將特徵圖的精華取出來，對圖壓縮又不失去太多資訊
- 常用手法有Max, Average, Sum

![](https://i.imgur.com/6tnaWWy.png)

- Pros:
    - 矩陣變小，資訊量幾乎不變
    - 增強圖像抗干擾能力(變形、扭曲)
    - 偵測不受該物體的絕對位置影響



#### 3. Fully-Connected Layer layer (全連接)
- 又稱全連接層、密集層、Dense layer, 可以對比於稀疏層

![](https://i.imgur.com/qZqZQTl.png)

## Multi-class classification with CNN
### Implement 實作
- 再一次以MNIST作為範例
- 上一次的Fully Connected NN的 accuracy約97-98%
- 簡單的CNN網絡會準確度會提升嗎?


### MNIST datasets
- Input: 手寫數字黑白(灰階)圖片
    - 28 pixel * 28 pixel 
    - ![](https://i.imgur.com/xxp1FxQ.png)
- Output: 數字類別
    - 2	(字元)
    - [0 0 1 0 0 0 0 0 0 0]	(One-hot vector)
- Data type: uint8 (代表可存放2^8的integer數字, 使每一灰階圖像的像素值落在0~255之間)
- 60,000 training set / 10,000 testing set
- 載入資料集
    - 4個Numpy array組成
    ```python
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    ```
- data preprocessing
    - reshape (np.array method)
    - 將3維(Dimention)的張量(tensor)，轉為2維(Dimention)的矩陣(matrix)
    - (即一張圖以一陣列vector表示)
    ```python
    train_images.shape
    # (60000, 28, 28)
    train_images = train_images.reshape((60000, 28 * 28))
    # (60000, 28 * 28)
    ```
    - astype (np.array method)
    - 轉換為float格式 (以計算/255的黑白顏色分布)
    ```python
    train_images = train_images.astype('float32') / 255
    ```

#### 建構一個CNN

```python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
- 說明
    - layers.Conv2D(32, (3, 3)
        - 32表示filter數量，(3, 3)為fileter長寬


#### 卷積後，利用FCNN傳遞給分類器
- 上一步最後一層為3D tensor(3, 3, 64)
- 利用**layer.Flatten()** 將3D展開為1D的(576, )
- 進而以FCNN傳遞到最後softmax分類器

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## Binary classification with CNN
### Dogs-vs-cats datasets
- 共25000張
- 前12500張是cats，後12500張是dogs
- 整理data，擷取較小的數據集後..
    - 貓狗各切半
    - training 2000 張 (1000/1000)
    - validation 1000 張 (500/500)
    - testing 1000張 (500/500)
- data preprocessing
    - 將JPG轉為RGB像素network
    ```python
    # 將所有圖像乘以 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    ```
    - 並將所有圖像都限制在 150 * 150
- 丟入fit，training完以後？

![](https://i.imgur.com/DFlBnfZ.png)

#### data augmentation
![](https://i.imgur.com/QyFZVhS.jpg)

## Reference
- https://github.com/exeex/ml-course
- https://medium.com/@CinnamonAITaiwan/深度學習-cnn原理-keras實現-432fd9ea4935


###### tags: `MachineLearning` `DeepLearning`


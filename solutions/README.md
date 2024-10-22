# 解答


*整体思路可以概括为：使用随机采样进行数据增强，解决样本不足问题；使用bagging，集成多个前馈神经网络进行预测。*

*Notebook中的数据处理部分包含了对更原始数据的预处理，使用时应做简单的更改，更改方法参考[model.py](/model/model.py)或[classifier.ipynb](/notebook/classifier.ipynb)。*


## 依赖
- scikit-learn
- tensorflow 2.x


## 分类任务

1. 对于每个布匹，随机不放回采样15个特征数组求平均，可以得到一个大小为(228, 3)的二维数组。对于每个布匹，这个采样过程进行2001次，即可扩大数据量2001倍。在继续后面的步骤前，将二维数组展平为一维数组。
2. 使用PCA降维为228维。
3. 训练15个前馈神经网络，对于每个网络，从训练集中随机放回采样90%的数据作为训练集。
4. 预测时，每个布匹同样采样2001次，并通过15个网络，投票决定最终预测结果。


## 回归任务

1. 对于每个布匹，随机不放回采样15个特征数组求平均，可以得到一个大小为(228, 3)的二维数组。对于每个布匹，这个采样过程进行1001次。与分类任务不同的地方是，数组每一列对应一类特征，3类特征分开处理，每类特征都是228维。
2. 使用PCA降维为228/2维（需要3个PCA，分别处理3类特征）。
3. 对于每一类特征，分别训练3个前馈神经网络，对于每个网络，从对应类别的训练集中随机放回采样50%的数据作为训练集。
4. 预测时，每个布匹同样采样1001次，每类特征通过对应PCA和3个网络，共得到1001*9个预测，将平均值作为最终结果。

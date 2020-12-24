# 训练

**目录结构1**
```
model
|- config.json
|- customize_service.py
|- model.py
```

1. 在OBS中建立目录，如**目录结构1**所示，保存[model](/model)中的模型配置文件[config.json](/model/config.json)、推理文件[customize_service.py](/model/customize_service.py)和模型/训练文件[model.py](/model/model.py)。
2. 创建算法，AI引擎选择`PyTorch-1.0.0-python3.6`，代码目录选择`/model`，启动文件选择`/model/model.py`，输入路径映射中配置`data-dir`映射到某个地址（该地址任意），输出路径映射中配置`model-dir`映射到某个地址（该地址任意）。
3. 为该算法创建训练作业，训练输入选择到包含`train.json`和`test.json`两个文件的OBS目录，模型输出设定为`/model`（即算法的代码目录）。
4. 等待训练完成，可以看到`/model`目录下增加了PyTorch模型文件`classifier.pth`和PCA模型文件`classifier-pca.pkl`，如**目录结构2**所示。


# 部署

**目录结构2**
```
model
|- classifier.pth
|- classifier-pca.pkl
|- config.json
|- customize_service.py
|- model.py
```

1. 导入模型，从OBS中选择元模型，路径为`/model`。
2. 等待导入完成，部署为在线服务。
3. 服务启动以后，请求json字符串是`{"sample": feat}`的格式，其中`feat`与训练/测试集中布匹的`feat`格式一样；返回的json字符串是`{"sample": name}`的格式，其中`name`与训练/测试集中布匹的`name`属性一样。


# API访问

API访问方法请参考官方文档，需要预先获取Token，并在请求头中添加该Token。

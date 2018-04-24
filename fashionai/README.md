Fashion AI 代码说明
====
    本md包含了Fashion AI服饰关键点识别的训练、测试方法描述和脚本使用说明.

## 依赖库说明
* TensorFlow 版本1.1及以上
* Numpy版本1.12及以上
* OpenCV版本2.4.10及以上

## 训练步骤说明
* ### 训练过程概述 <br>
    本代码采用了[Realtime multi-person](http://xueshu.baidu.com/s?wd=paperuri%3A%28ee7d699fb12eb95daec96f29da5452b9%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1611.08050&ie=utf-8&sc_us=3215272012347045556)论文提出的人体关键点检测网络结构，并进行了改进，保留了前2个stage以提升训练速度,同时精度损失不大。在现有的磁盘文件格式的数据集上直接进行训练。首先从磁盘读取一个batch的数据，对每个样本进行数据增强后，对CNN进行正向和反向运算。对2个stage均采用L2损失函数计算loss。迭代30000次，学习率初始化为2e-5，进行指数衰减，分别在迭代次数达到20000和25000时衰减0.1
    
* ### 训练代码使用 <br>
    训练代码全部包含在train_cpm.py文件中。训练步骤如下：
    - #### 数据集 <br>
        将官方数据集解压得到train文件夹，下面包含Annotations和Images两个文件夹。其中Annotations文件夹下面包含一个train.csv文件，即标签文件；Images文件夹下面包含5种服饰类别，每个文件夹下面包含个类别的训练样本
    - #### 训练 <br>
        打开`train_cpm.py`，将dataset_root改为Images文件夹的路径，并在DataLoader的构造函数中为其指定csv_path，即train.csv的路径:
        ```
        loader=loader.DataLoader(csv_path=op.join(dataset_root,'Annotations/train.csv'),dataset_root=dataset_root)
        ```
        在第17~21行修改迭代次数、初始学习率、衰减倍数和触发衰减步骤数等配置参数。执行train_cpm.py即可开始训练。由于本版本没有采用多GPU训练和TfRecord格式的数据，训练较慢

## 测试步骤说明 <br>
测试代码包含 `test.py`, `test_score.py`和`gen_test_csv.py`三个文件,功能如下所述：
* #### `test.py`
    该脚本每次测试一张图像，并将关键点的检测结果绘制到原图上进行展示。也可以对一个已经完成的测试csv文件进行检测，通过按键控制显示csv中每一个图像的可视化展示，从视觉角度对检测结果进行评估
* #### `test_score.py`
    该脚本用于在验证集上对模型精度进行验证。本次在warm_up_train_20180222训练数据集上进行了验证，用于在提交结果之前在本地先对模型精度进行评价。和train_cpm.py类似，需要传入Images文件夹的路径和csv文件的路径：
    ```
    image_root='/home/XXX/fashionAI_warm_up_train_20180222/train/'
    
    csv_name='/home/XXX/fashionAI_warm_up_train_20180222/train/Annotations/annotations.csv'
    ```
* #### `gen_test_csv.py`
    该脚本用于产生测试数据集的csv文件。需要手动修改脚本的4个参数：<br>
    测试数据集的Images文件夹的路径 <br>
    测试数据集的文件列表，即test.csv <br>
    训练数据集的标注文件train.csv，用于获得完整的title <br>
    训练好的检查点文件
    ```
    dataset_root='/home/XXX/test/'
    model_loader=ml.ModelLoader(load_mode='ckpt',model_path='models/fashion.ckpt-%d'%iteration)
    model_loader.load_model(session=sess)
    
    with open('/home/XXX/Annotations/train.csv','r') as tf:
        reader=csv.reader(tf)
        header=None
        for row in reader:
            header=list(row)
            break
    ```
    执行该脚本即可在当前目录下生成test.csv

# RetinaNet: Focal Loss for Object Detection

**动机：**

​	单阶段目标检测过程中，一张图里面的目标比较少，但是采样很多，这就造成正负样本不平衡，负样本过多，导致最终的loss受负样本的影响极其大，而正样本在loss中占据的比例特别小，导致模型能学习到正样本的信息少，容易被负样本主导。

**相关知识积累：**

<img src=".\imgs\image-20250806162622869.png" alt="image-20250806162622869" style="zoom:50%;" />

- easy negative：全是背景，比较容易判断的负样本
- easy positive：全是物体，比好容易判断的正样本
- hard negative：包含部分物体，但大部分为背景，比较难判断的负样本
- hard positive：包含部分背景，但大部分为物体，比较难判断的正样本
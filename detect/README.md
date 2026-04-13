# yolov5

## 主要结构

骨干网络（Backbone）：CSP-Darknet53  
颈部网络（neck）：SPPF, CSP-PAN  
头部网络（head）

## 图像预处理部分

### Mosaic 数据增强 ：

将四张图像拼接到一起，形成一张图像，增加数据的多样性。先随机抽取四张图像，并按照等比列将最大边缩放到指定的size（一般是640），然后将四张图像拼接成一个（2*size）*（2*size）的一张图，最后再~~（缩放）~~裁剪到指定size。 


### 自适应锚框计算：

由于数据集不同，yolov5给的锚框不一定适合自己的数据集，所以这个时候可以启动自动锚框计算功能，它通过获取数据集中所有目标的宽和高，然后只要通过k-means的方法进行聚类，来计算合适的锚框大小。

### letterbox：

一般在图像预处理的阶段，使用resize函数将图像统一缩放到一个指定的大小，通常原始图像不是正方形的，但是网络的输入一般都是正方形的，直接进行resize则会将图像拉伸变形，yolov5的自适应缩放是先将图像按照等比列的方式缩放，将图像的最大边缩放到指定的size，使用指定像素填充短边，使其达到指定size。

注：指定的size必须要满足32的倍数，因为yolov5需要进行五次下采样。

## 训练

### 训练相关的参数

	# rect：矩形训练，默认关闭,加了default =True
	# resume：断点续训，如果训练过程中意外断掉，或者是想接着训练，就可以使用该参数
	  使用方法：
	  1、对于被打断，继续训练，使用命令行输入：
	  python train.py --resume ./run/train/exp/weights/last.pt
	  2、对于训练完成之后，想继续训练，需要修改想要继续训练的文件夹的opt.yaml里面的epochs参数，修改为想要继续的轮数，比如说之前模型训练了100轮，现在想加50轮，就可以把epochs修改为150。然后找到./utils/torch_utils.py里面的smart_resume函数，将start_epoch修改之前的训练轮数，比如100。
	  然后使用命令：
	  python train.py --resume ./run/train/exp/weights/last.pt
	# noautoanchor：关闭自动计算锚框功能，默认关闭
	# noplots：不保存可视化文件
	# evolve：使用超参数优化算法进行自动调参，默认关闭
	# cache：缓存数据集，以加快训练速度
	# image-weights: 对于那些训练不好的图片，会在下一轮中增加一些权重
	# multi - scale: 是否进行多尺度训练，一般设置几种不同尺度的图片，训练时每隔一定iterations随机选取一种尺度训练
	# single-cls: 数据集是否只有一个类别，默认False
	# cos-lr：用于对学习速率进行调整，默认为 False，（通过余弦函数来降低学习率）
	# label-smoothing: 对标签进行平滑处理,防止过拟合

### 网络的结构

### 头部

#### 标签分配：

1、跨anchor检测

具体的做法是将真实的标签复制三份，分别去和三层anchor计算宽高比，ratio1= 真实标签的宽高/anchor宽高，ratio2 = anchor宽高/真实标签的宽高，然后取两者中的最大值和设定阈值进行比较，如果比值太大的话，说明目标不适合用这个anchor进行检测，这一步是为了匹配合适的anchor。

2、跨grid检测

在经过跨anchor检测之后，可以匹配到合适的anchor，那么此时，这个真实的标签一定落在了某个grid内，这个grid有左，上，右，下四个邻域的grid，然后根据真是标签的中心位置，将离的更近的两个邻域grid也作为预测网格，也就是一个真实的标签（即一个目标对象）可以由三个grid预测，这是为了解决，目标太大跨grid的问题。

3、跨分支预测：

假设一个GT框可以和2个甚至3个预测分支上的anchor匹配，则这2个或3个预测分支都可以预测该GT框。即一个GT框可以在3个预测分支上匹配正样本，在每一个分支上重复anchor匹配和grid匹配的步骤，最终可以得到某个GT 匹配到的所有正样本。

#### 损失：

1、分类损失：BCE
2、bounding box的损失：  
（1）IOU_Loss：预测框与GT框之间的交集/预测框与GT框之间的并集  
（2）GIOU_loss：引入最小外接矩形。1-（IOU-(最小外接矩形的面积-预测框与GT框之间的并集)/最小外接矩形的面积）   
（3）NMS非极大值抑制：  
   a）对所有预测框的置信度降序排序  
   b）选出置信度最高的预测框，确认其为正确预测，并计算他与其他预测框的IOU   
   c) IOU>threshold阈值就直接删除
   d) 返回步骤a

## 推理时的相关参数

	# conf-thres：检测时置信度阈值，低于这个置信度阈值的直接被筛除
	# iou-thres：iou阈值，当预测框与真实框的重合度大于这个阈值是才被认为是正样本
	# max-det：设置一张图像中的最大检测数量，默认是最多检测 1000 个目标
	# view-img：是否在检测的过程中展示检测的结果
	# save-txt：是否把检测结果保存成一个.txt的格式，txt文件里面保存了一些类别信息和边框的坐标信息。
	# save-conf：以.txt的格式保存目标的置信度，一般配合 - -save - txt一起使用
	# save-crop：是否把模型检测的物体裁剪下来。裁剪的物体图片会保存在crops目录下。
	# nosave：不保存预测的结果。
	# classes：检测的类别，可以设置一个或者多个类别，根据自己的类别名去设置。比如coco数据集里面0，代表person。
	# agnostic-nms：跨类别nms，比如待检测图像中有一个长得很像篮球的足球，pt文件的分类中有足球和篮球两种，那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是篮球。开启agnostic-nms后，那只会框出一个框。
	# augment：推理的时候是否开启图像增强
	# visualize：特征图可视化，开启之后，可以将每层的可视化已图片的形式保存下来
	# update：在模型训练的最后阶段去除优化器信息，以减小模型文件的大小，并将模型准备好用于推断或其他目的。





# yolov6

yolov6主要是使用重新参数化的方法，训练和推理的时候使用不同的模型结构，从而在不影响训练的情况下，加速推理的速度。

## backbone

yolov6训练时的主干网络主要是repvgg的结构模块，具体来说，每一个小的模块都是由3X3卷积、1X1卷积以及恒等映射组成的，并行多个分子和一般能增加模型的表征能力，而在推理的阶段，为了加快推理速度，将这三者融合成一个3X3的卷积。

## RepPANNeck

具体来说就是将CSP-PAN里面的CSP模块替换成repvgg的模块，整体结构和v5没有什么差别。


## head部分

yolov5的检测头部分，分类和回归是耦合在一起的，使用相同的特征来同时预测目标的类别和位置这种方式可能会导致任务之间的干扰，v6使用和yolovx使用相同的方式。优势如下：  
（1）将检测和分类的任务解耦，使用两个分支去分别预测类别和位置，使得每个分支能够专注自己的任务，不至于混淆。  
（2）而且解耦头的设计使得模型有更高的灵活性，可以针对不同任务设计不同的损失函数和训练策略。

## 锚框

yolov5是基于anchor的方法，yolov5可以加载训练的标签，通过聚类来计算出最佳的anchor组合。而yolov6是使用anchor-free的方式。  

anchor-free：  

anchor point：是featurer map中每个像素点的中心坐标  
预测框数量 = anchor point数量 = 80 * 80 + 40 * 40 + 20 * 20  
即每一个anchor point对应一个anchor

## 正样本匹配

将三层的feature map映射到原图上，标签框内框到所有点都作为正样本，其他的都作为负样本，当检测目标出现重叠情况时，两个真实框框到的anchor point是有重叠的，那么这部分重叠的anchor point对应的anchor负责哪个目标的检测呢？这里需要计算anchor和真实框的iou值，通过比较anchor与真实框的重叠面积大小来判断这个anchor应该负责哪个目标的检测。





# yolov8

yolov8主要借鉴了yolov5，yolov6，yoloX等模型的设计优点，偏重在工程实践上上。

## backbone

同样是借鉴了CSP模块的思想，将yolov5中的C3替换成了C2f模块，C3是将输入特征并行的进行两次卷积，每个卷积的输出是目标输出的一半，其中一部分的卷积再经过n次Bottleneck之后与另外一半的特征进行concat，然后共同再进行一次卷积，一共进行了三次卷积操作。C2f先通过一个卷积，然后将卷积之后的特征split，一部分经过n次Bottleneck之后再和另外一部分的卷积concat，最后再卷积一次，一共执行了两次卷积。

## neck

继续使用PAN的思想

## head

和yolov6相似，yolov8的head部分使用解耦头结构，将分类头和检测头分开，不再使用anchor-base，和v6一样使用anchor-free。

## loss

#### 正负样本分配策略：

Task-Aligned Assigner：

简单来说就是针对所有像素点预测的cls score和reg score，通过加权的方式得到最终的加权分数，通过加权分数进行排序后，选择topk个样本。

处理流程：

（1）解析预测框坐标（l,t,r,b）-> (xmin, ymin, xmax, ymax)

（2）粗筛正样本：

​	筛选出落在gt_box范围内的anchor，作为正样本

​	这里是将每层feature map中的像素映射到原图中，去寻找真实框内的anchor point

（3）在上一步粗筛出来的正样本基础上精筛正样本：

- 提取bbox_score， 计算CIoU
- 计算出的align_metric:(TAL)    
- 根据align_metric的值，筛选出top-10作为正样本

（4）处理一个anchor point可能匹配到多个gt_box的情况，仅保留最大CIoU的那个匹配

#### loss计算：

yolov8没有了之前的object分支，只有分类分支和检测分支。分类分支使用BCE Loss，检测头的回归损失函数使用的是CIou_Loss和Distribution Focal Loss。

## anchor

​		基于anchor的方法训练目标是回归anchor与真实框（Ground Truth）之间的差值，而anchor-free方法的训练目标是通过关键点（角点或者中心）直接回归GT的值。

### 1、anchor-base

主要是anchor的宽和高，不需要考虑anchor的位置，因为anchor的位置是feature map每个像素的位置，yolov5一共有三个feature map，每一层的feature map中的每一个像素给定3个固定大小的anchor，所以yolov5一共有9个anchor。

yolov5给定的anchors大小是相对于640*640训练尺寸的，这里就需要注意，如果我们把训练尺寸改变的很多，anchors的尺寸可能会需要修改，但是如果自动计算的参数打开，会自动计算最合适的anchors尺寸。

### 2、anchor-free

之前的anchor-base需要设计anchor，主要是设计anchor的宽和高

anchor-point（anchor的中心点坐标：feature map中每个像素的坐标，每个像素对应一个anchor，anchor的宽和高是不确定的）

yolov8预测框输出的是（l，t， r， b）代表中心坐标到预测框四个边的距离。





# yolov8-obb

标注工具：rolabelImg

**标注标签格式**：边框的中心点坐标（cx，cy），图像的宽w，高h，框**顺时针**旋转的角度angle（范围为（0，$\pi$））

**训练标签格式**：四个顶点的坐标

**转换过程**：

二维坐标下，（x，y）以原点为中心**逆时针**旋转$\theta$之后得到的（x'，y'）的坐标计算公式为：
$$
x' = xcos(\theta)-ysin(\theta)
$$

$$
y' = xsin(\theta)+ycos(\theta)
$$

而rolabelme是以**顺时针**为正向，需要将$-\theta$带入到公式中，调整公式如下：
$$
x' = xcos(\theta) + ysin(\theta)
$$

$$
y' = -xsin(\theta) + ycos(\theta)
$$

且输入的角度依然是**逆时针**旋转的角度。

OBB任务只是在detect任务的基础上多了一个旋转角度的输出头。





# yolov9

主要研究了数据通过深度网络传输时数据丢失的问题（信息瓶颈和可逆函数）

提出的方法：

1、设计了PGI辅助可逆分支，其实就是一个并行的backbone+辅助分支，相当于把没有经过PAN单独使用卷积提取的特征和经过PAN处理的特征做一个融合，提供更多的先验知识。





# yolov10

主要解决了NMS实时推理的耗时问题。

one-to-many label assign : 意味着一个真实目标会被分配给多个候选区域作为正样本进行学习。

one-to-one label assign : 意味着一个真实目标只会被分配给一个候选区域作为正样本学习。

比如说，基于iou的时候，一张图像有一个物体的真实框，有10个anchor box，其中有6个与真实框的iou都大于0.5，那么在one-to-many label assign下，这6个anchor boxes都会被分配给这个物体作为正样本，而one-to-one label assign只会选择iou最大的那个anchor作为正样本。



## dual label assignmets策略

yolov10在训练的时候包含两个结构相同，但是参数不同的head，一个是one-to-many，一个是one-to-one，训练的时候两个head同时，而推理的时候只需要使用one-to-one的head，推理的时候不需要nms。



# yolov11

## backbone

改进了backbone，设计了C3k2机制，在网络的浅层，c3k参数设置为False，等于v8中的C2f，当c3k参数设置为True时，相当于在v8的bottleneck里面又套了一个循环，又有若干个bottleneck，整体模型的复杂度提高了。

## neck

就是在PAN的部分设计了一个PSABlock的模块，这个模块里面添加了多头注意力。

## head

相较于v8来说，v11在分类分支部分，将卷积换成了深度可分离卷积，来提高检测头的速度。



# DETR

## 1、基础知识

**NMS**（非极大抑制）：在一般的目标检测算法里面都会用到NMS来进行后处理，NMS主要是为了去除同一个目标的多个冗余的预测框，它通过计算两个框的交叠面积来判断预测的是不是同一个目标。

**缺陷**：只使用了检测框的信息，不使用图像特征，如果两个目标离的很近的话，就会判别成一个目标。

**原因**：NMS在对每个anchor进行回归的时候是独立进行的两个anchor之间没有信息交换

**解决思路**：让两个anchor之间有信息交换，让一个anchor知道另外一个anchor已经去预测某一个目标了，从而避免预测重复的冗余框。

**related work**：CNN+LSTM

思路：

​       先使用CNN提取特征，假设经过CNN提取的特征维度为：14x14x1024，那么将每一个空间位置的1x1x1024维度的特征向量并行送入到LSTM中。LSTM的每一个step都是基于已有的结果进行输出的，如果是在上个step已经预测出了某个ground truth，那么后面的step将不会对这个ground true重复预测，通过RNN的前后依赖性，解决预测框之间相互独立的问题。简单来说就是上面一个step检测出来一个目标，在已经的信息记录下，后面的step就不会再去检测这个目标。





# YOLO-World

开放集检测：检测参与训练的类别

### 一、YOLO-World整体结构：

<img src="\imgs\image-20251024112712798.png" alt="image-20251024112712798" style="zoom:70%;" />



**标注信息**：传统目标检测是位置框+类别

​                   YOLO-World是区域-文本对（文本可以是类别名字，可以是一组短语，也可以是一段描述）

**模型输入**：图像+文本

**模型输出**：预测框以及对应物体的嵌入向量

**模型框架**：

- YOLO Detector（基于yolov8）：用于提取图像的多尺度特征信息，Darknet骨干网络+PAN+边界框回归和对象嵌入head。

- Text Encoder：输入文本被YOLO-Wrold的文本编码器（CLIP预训练模型）编码成文本嵌入
  $$
  W = TextEncoder(T)\in R^{CXD}
  $$
  C是名词数量，D是嵌入维度，当输入的文本为标题或指代表达时，采用简单的n-gram算法提取名词短语，再将其输入到文本编译器。

- Text Contrastive Head：解耦头设计，使用两个3X3的卷积分别回归预测框和物体嵌入向量。然后计算物体嵌入向量和文本嵌入向量的相似度：
  $$
  s_{k,j} = \alpha . L2\_Norm(e_{k}).L2\_Norm(w_{j}) + \beta
  $$
  其中$e_{k}$是目标嵌入向量，$w_{j}$表示第 j 个文本嵌入

- Training with Online Vocabulary：在训练过程中，一个mosaic图像中的四张图中包含的目标文本标签为正面名词，除此之外，再随机抽取一些不在这个mosaic图像里面的类别名词作为负面名词，使模型能够聚焦到当前样本里的物体同事通过负样本扩大见识，提升对各类别的识别能力。

- Inference with Offline Vocabulary：先给提示后进行检测，可以自定义提示，yolo-world将这些提示编码成离线词汇嵌入向量，避免每次输入的计算开销。

**Re-parameterizable Vision-Language PAN**

<img src=".\imgs\image-20251027100319467.png" alt="image-20251027100319467" style="zoom:60%;" />

- Text-guided CSPLayer ： 

通过下面的公式，融合文本特征和多尺度图像特征：

<img src="\imgs\image-20251027143030429.png" alt="image-20251027143030429" style="zoom:60%;" />

其中$X_{l}^{'} = X_{l}*\delta(max_{j \in \{1..C\}}(X_{l}W_{j}^{T})^{T}$

- **Image-Pooling Attention**:简单来说就是在多尺度特征上使用最大池化操作来聚合图像特征。



# yolo26

#### 创新点：

- 移除DFL（分布焦点损失），简化了边界框预测逻辑，提升了硬件兼容性，解决了超大目标检测的可靠性问题。
- 原生的端对端模型，直接生成预测结果，无需非极大抑制。
- 引入MuSGD优化器，增强稳定性，收敛更快。
- ProgLoss+STAL，提升了小目标检测的精度。
- CPU推理速度提升了高达43%。



#### 为什么训练的时候要使用两个检测头，以及为什么推理的时候只是用one-to-one的检测头就能够达到不错的效果？

​		one-to-many label assign需要NMS进行后处理，影响部署的速度，one-to-one label assign会带来额外的推理开销且效果不好。在训练的过程中会使用one-to-many的检测头来指导one-to-one的检测头，先分别计算损失，然后使用1-Wasserstein distance来最小化两个检测头的损失距离，从而使one-to-one的分类效果逼近one-to-many。

```
one-to-many：
​ 一个真实目标会被分配给多个预测框作为正样本，用来提供丰富的监督信息，提高模型的检测能力。

one-to-one：
​ 一个真实目标只会分配给一个预测框作为正样本，其他的预测框都被当作成负样本，从来达到在训练的过程中就抑制了多个预测框出现的情况。
（同yolov10）
```



# AFSS

```
相关工作：
在不改变检测器结构的前提下提升目标检测器的训练效率，研究方向可分为三类：
1、curriculum and self-paced learning：按照样本难度对数据进行组织，逐步引入难度更高的样本，但这种方法近优化了学习顺序，无法保证模型持续接触高信息量样本。
2、dataset pruning：利用早期损失或梯度统计量剔除冗余或者低价值图像来实现训练加速，但图像一旦剔除就无法再次参与训练。
3、dataset distillation：通过合成少量人工图像来近似原始数据集的分布特性，从而进一步降低训练开销，但该范式需要借助复杂的双层优化与生成模型进行额外的数据合成。
```

#### 方法：

第 t 轮的迭代流程：

​	根据 t-1 轮学习的准确度与召回率评估每张训练图像，分成简单、中等、困难三个等级

- 简单等级以最低频率参与训练，并周期性回顾，以避免灾难性遗忘
- 中等图像以中等频率参与训练，保持短期全覆盖，确保每张图像至少3轮内重新出现一次
- 困难样本全程参与训练

筛选出来的图像参与第 t 轮训练。

#### 整体结构：

![image-20260413160654682](.\imgs\image-20260413160654682.png)

#### 学习充分度指标：

$$
I_{i} = min (P_{i}, R_{i})
$$

根据学习充分度指标将数据分为三个等级：
$$
\begin{cases}
\text{Easy}, & \text{if } \min(P_i, R_i) > 0.85, \text{   2% 采样率 }\\
\text{Moderate}, & \text{if } 0.55 \leq \min(P_i, R_i) \leq 0.85, \text{   40% 采样率 }\\
\text{Hard}, & \text{if } \min(P_i, R_i) < 0.55， \text{   100% 参与训练 }
\end{cases}
$$

#### 持续复习机制（简单样本）：

对于简单样本，每次仅采样2%，为了防止遗忘，AFSS引入持续复习机制：

1、强制学习：超过10个epoch未被使用的Easy样本优先被召回参与训练

2、随即多样性：剩余的配额从剩余的简单样本中随机采样，保证多样性，补齐2%

3、总量约束：强制学习的部分不能超过Easy采样总量的一半，也就是不能超过简单样本总量的1%

#### 短期覆盖机制（中等图像样本）：

1、连续2个epoch未使用的Moderate样本强制选入下一轮训练，确保每张图像至少3个epoch出现一次

2、剩余的配额，从Moderate随机抽取补齐40%

#### 周期性状态更新：

每张图片的Precision、Recall和上次使用时间构成一个状态字典，每5个epoch更新一次，重新划分一个简单、中等、困难的等级。



























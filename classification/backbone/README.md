# 2025

## LSNet

LSNet: See Large, Focus Small

paper: https://arxiv.org/abs/2503.23135

**动机**：	

​		现在的网络框架一共有两种，一种是CNN一种是自注意力（代表Transformer），两种框架的缺陷：

<img src=".\imgs_backbone\selfAttention_conv.png" alt="selfAttention_conv" style="zoom:33%;" />

​		自注意力：第一，自注意力机制往往会过度关注缺乏显著相互关联的区域，导致非关键信息的聚合（例如再信息量较少的背景区域），简而言之就是会对背景这种信息量少的区域给予了过多的关注。第二，自注意力的计算复杂度高。两者阻碍了对低计算量下的高表征能力的追求。

​		CNN：卷积计算高效，但是卷积是由固定的内核去提取特征，因此对 不同上下文邻域缺乏敏感度，限制了轻量级模型的表达能力。

​		**LS convolution（论文提出的方法）**：采用大核静态卷积进行大面积感知，小核动态卷积进行小区域聚合。

**方法：**

​		<img src=".\imgs_backbone\LSNetConv.png" alt="LSNetConv" style="zoom:45%;" />

我的理解是橙色代表提取的特征（感知P），蓝色代表特征聚合（聚合A）的过程。

LS convolution特征如下：

（1）感知P和聚合A采用不同的上下文范围，利用一尺度上下文信息，同时捕获整体语境和细粒度细节。

（2）对于大空间范围的感知，使用高效的大核深度卷积，以最小开销扩展感知上下文。

（3）小范围邻域的聚合，采用子使用加权特征求和，其有限的聚合范围确保低计算成本下的效率，同时缓解自注意力中次要聚合的问题。

**整体架构**

![LSNet](.\imgs_backbone\LSNet.png)

- 第一步：Large-kernel perception，结合图示非常容易理解，首先使有和1x1卷积进行通道降维，然后用大核卷积获得大感受野的空间上下文信息，最后再通道升维。
- 第二步： Small kernel aggregation，本质是分组动态卷积。特征分为G组，每组使用的卷积核是Large-kernel perception生成的。这样可以有效地表示自适应细粒度特征，使模型对不同环境下的动态复杂变化敏感。



# 2024

## mobilenetV4





# 2020

## GhostNetV1

​		作者提出，在一个训练好的神经网络中，会存在很多的冗余特征，而这些冗余的特征可以通过一些廉价的操作从别的特征图变换中得到。因此作者的想法是，通过卷积得到一部分的特征，然后通过一些廉价的操作从这些特征图中得到冗余的特征。ghost卷积的结构如下：

<img src=".\imgs_backbone\ghost.png" alt="image-20250214165043231" style="zoom:50%;" />

卷积过程：

（1）先通过卷积生成一部分的特征图。

（2）对每个通道通过廉价的线性变换（其实就是对每个通道进行一次卷积，等价于深度卷积）。

（3）将第一步生成的特征图与第二步生成的特征图拼接到一起。

ghost bottleneck:

<img src=".\imgs_backbone\ghost_bottleneck.png" alt="image-20250214165338225" style="zoom:50%;" />

（1）stride = 1的情况：

      当stride=1的时候，不进行下采样，特征图的宽高保持不变。这一部分思想类似mobilenetV2，先通过一个Ghost module用作扩展层，增加特征的通道数，扩展出冗余的特征，这是为了防止低维度的特征在ReLU激活的时候造成信息丢失。经过ReLU激活之后使用一个Ghost module降低特征维度，减少计算量。

（2）stride = 2的情况：

      当stride=2的时候，对特征进行下采样。两个Ghost module作用和stride=1的时候是相同的，除此之外在两个Ghost module之间加入一个stride=2的DWConv，实现下采样的功能。





# 2019

## mobilenetV3

mobilenetV3整体还是采用了v2的结构，主要改进点在于：

1、使用NAS执行模块级搜索，构建一个高速的框架。

2、加入了SEAttention。

3、将最后一步的平均池化层前移并移除最后一个卷积层，引入h-swish激活函数。

![v3](.\imgs_backbone\mobilenetv3.png)



# 2018

## mobilenetV2

论文地址：https://arxiv.org/pdf/1801.04381

在mobilenetV1的基础上，增加了倒置残差结构（Inverted Residuals）和线性瓶颈层（Linear Bottlenecks）。

1、倒置残差结构：

由于mobilenet使用的是深度可分离卷积，DWConv每个卷积核的通道数都是1，就是导致特征通道数太少，不利于提取丰富的特征信息，于是V2设计了倒置残差结构。与ResNet的残差结构不同，Inverted Residuals输入和输出的通道数很少，在中间对卷积通道数进行扩展。

![Inverted_Residuals](.\imgs_backbone\Inverted_Residuals.png)

在Inverted_Residuals中的扩张层和深度可分离卷积之后，使用的激活函数是ReLU6。Inverted Residuals 具体的实现如下：

<img src=".\imgs_backbone\mobileV2.png" alt="mobile" style="zoom:50%;" />



2、线性瓶颈层

如上图所示，在倒置残差结构里面的最后一层没有使用ReLU激活函数，还是使用的是一个1x1的线性层，防止非线性操作破坏过多的信息，避免信息损失。



# 2017

## mobilenetV1

论文地址：https://arxiv.org/pdf/1704.04861

主要是使用深度可分离卷积

深度可分离卷积包含两个过程：逐通道卷积（depthwise convolution）和逐点卷积（pointwise convolution）

DWConv:

<img src=".\imgs_backbone\DWConv.jpg" alt="dwconv" style="zoom:45%;" />

PWConv:

<img src=".\imgs_backbone\PWConv.jpg" alt="pwconv" style="zoom:40%;" />

1、逐通道卷积：

逐通道卷积是一个卷积核负责一个通道，比如说，现在有一个28X28X256 的 feature map，那么逐通道卷积就一共有256个卷积核，每一个卷积核负责一个通道，经过逐通道卷积之后得到的 feature map 和输入的 feature map 的通道数是一样的，没有改变。

2、逐点卷积：

前面的逐通道卷积并不改变特征的通道数，所以无法拓展 feature map，而且由于每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的特征信息，因此需要逐点卷积来进行空间信息上的组合并且扩展特征的通道维度。

逐点卷积的每个卷积核的通道数和输入特征的通道数是一样的，但是卷积核的宽高都为1，也就是1X1的卷积，也就是将特征在通道方向上进行加权组合。例如，现在有一个28X28X256的特征，那么逐点卷积的一个卷积核的大小是1X1X256，卷积核的个数可有网络结构设计而定。




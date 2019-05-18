# TensorMask-Review
Sliding-window object detectors that generate boundingbox object predictions over a dense, regular grid have advanced rapidly and proven popular. In contrast, modern instance segmentation approaches are dominated by methods that first detect object bounding boxes, and then crop and segment these regions, as popularized by Mask R-CNN. In this work, we investigate the paradigm of dense slidingwindow instance segmentation, which is surprisingly underexplored. Our core observation is that this task is fundamentally different than other dense prediction tasks such as semantic segmentation or bounding-box object detection, as the output at every spatial location is itself a geometric structure with its own spatial dimensions. To formalize this, we treat dense instance segmentation as a prediction task over 4D tensors and present a general framework called TensorMask that explicitly captures this geometry and enables novel operators on 4D tensors. We demonstrate that the tensor view leads to large gains over baselines that ignore this structure, and leads to results comparable to Mask R-CNN. These promising results suggest that TensorMask can serve as a foundation for novel advances in dense mask prediction and a more complete understanding of the task. Code will be made available.

## TensorMask
https://arxiv.org/pdf/1903.12174.pdf 2019

## RCNN
https://arxiv.org/pdf/1311.2524v3.pdf 2014

## Fast RCNN
https://arxiv.org/pdf/1504.08083.pdf 2015

## Faster RCNN
http://de.arxiv.org/pdf/1506.01497 2016

## A-Fast-RCNN
https://arxiv.org/pdf/1704.03414.pdf 2017

## Mask RCNN
https://arxiv.org/pdf/1703.06870v1.pdf 2018

# RCNN-----Rich feature hierarchies for accurate object detection and semantic segmentation Tech report (v3)（开源）

https://github.com/rbgirshick/rcnn 2014年

在PASCAL VOC标准数据集上测量的目标检测性能在最近几年趋于稳定。性能最好的方法是复杂的、可理解的系统，这些系统通常将多个底层图像特性与高层上下文结合起来。在本文中，我们提出了一种简单、可扩展的检测算法，相对于VOC 2012年的最佳检测结果，证明平均精度(mAP)在30%以上，达到了53.3%。我们的方法结合了两个关键的见解:

(1)可以将大容量卷积神经网络(CNNs)应用于自底向上的区域建议中，以对目标进行定位和分割;

(2)由于我们将区域建议与CNNs相结合，我们将我们的方法称为R-CNN:具有CNN特性的区域。我们还将R-CNN与OverFeat进行了比较，OverFeat是最近提出的一种基于类似CNN架构的滑动窗口检测器。我们发现R-CNN在性能上远远超过了OverFeat：http://www.cs.berkeley.edu/ ˜ rbg/rcnn.

Ross Girshick (rbg)主页：http://www.rossgirshick.info/

# Fast R-CNN（开源）

本文提出了一种快速的基于区域的卷积算法用于目标检测的网络方法(快速R-CNN)。快R-CNN基于之前的工作，使用深度卷积网络对目标提案进行有效分类。与以前的工作相比，Fast R-CNN采用了几个更新来提高训练和测试速度，同时也提高了检测精度。Fast R-CNN训练非常深的vgg16网络，速度9×比R-CNN快，测试时速度213×，在PASCAL VOC上实现了更高的mAP2012. 与SPPnet相比，快速R-CNN训练VGG16 3×更快，测试速度10×更快，更准确。Fast R-CNN是用Python和c++(使用Caffe)实现的，在https的开源MIT许可下可用

https://github.com/rbgirshick/fast-rcnn 2015年

我们提出了一种新的训练算法，在提高R-CNN和SPPnet速度和精度的同时，弥补了R-CNN和SPPnet的不足。我们称这种方法为Fast R-CNN be-因为它训练和测试相对较快。

# A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection(开源)

https://github.com/xiaolonw/adversarial-frcnn 2017年

How do we learn an object detector that is invariant to occlusions and deformations? Our current solution is to use a data-driven strategy – collect large-scale datasets which have object instances under different conditions. The hope is that the final classifier can use these examples to learn invariances. But is it really possible to see all the occlusions in a dataset? We argue that like categories, occlusions and object deformations also follow a long-tail. Some occlusions and deformations are so rare that they hardly happen; yet we want to learn a model invariant to such occurrences. In this paper, we propose an alternative solution. We propose to learn an adversarial network that generates examples with occlusions and deformations. The goal of the adversary is to generate examples that are difficult for the object detector to classify. In our framework both the original detector and adversary are learned in a joint manner. Our experimental results indicate a 2.3% mAP boost on VOC07 and a 2.6% mAP boost on VOC2012 object detection challenge compared to the Fast-RCNN pipeline. We also release the code 1 for this paper.

# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（开源）

https://github.com/rbgirshick/fast-rcnn 2016年

最先进的目标检测网络依赖于区域建议算法来假设目标位置。SPPnet[1]和快速R-CNN[2]等技术的进步，降低了检测网络的运行时间，暴露了区域提案计算的瓶颈。在这项工作中，我们引入了一个与检测网络共享全图像卷积特性的区域建议网络(RPN)，从而实现了几乎免费的区域建议。RPN是一个完全卷积的网络，它同时预测每个位置的对象边界和对象得分。对RPN进行端到端训练，生成高质量的区域建议，快速R-CNN对其进行检测。我们进一步合并RPN和快速的R-CNN成一个单一的网络，通过共享他们的卷积特征-使用最近流行的术语神经网络与“注意”机制，RPN组件告诉统一网络在哪里查找。对于非常深的VGG-16型号[3]，我们的检测系统在GPU上的帧率为5fps(包括所有步骤)，同时在PASCAL VOC 2007、2012和MS COCO数据集上实现了最先进的目标检测精度，每张图像只有300个提案。在ILSVRC和COCO中2015年的比赛，更快的R-CNN和RPN是在几个轨道上获得第一名的基础。代码已经公开。

# MaskR-CNN（开源）

https://github.com/matterport/Mask_RCNN  2018年

我们提出了一个概念简单、灵活和通用的对象实例分割框架。我们的方法可以有效地检测图像中的目标，同时为每个目标生成高质量的分割掩码。该方法称为Mask R-CNN，扩展速度更快 R-CNN通过添加一个分支来预测一个对象掩码，该分支与现有的用于包围框识别的分支并行。蒙版R-CNN训练简单，只增加了一个小开销到更快的R-CNN，运行在5帧每秒。此外,掩模R-CNN很容易推广到其他任务中，例如降低我们在相同框架下估计人体姿态。我们展示了COCO套件中所有三个方面的顶级结果，包括实例分割、包围框对象检测和人员关键点检测。有了- out铃声和哨声，面具R-CNN在每个任务上都优于所有的前，单模型的条目，包括可可2016挑战赛冠军。我们希望我们的简单而有效的方法将作为一个坚实的基线，并有助于简化未来在实例级识别方面的研究。

https://github.com/facebookresearch/Detectron

# TensorMask: A Foundation for Dense Object Segmentation（未开源） 2019

Xinlei Chen   http://xinleic.xyz/#

Ross Girshick    http://www.rossgirshick.info/

Kaiming He   http://kaiminghe.com/

Piotr Dollár  http://pdollar.github.io/

在目标检测任务中，采用滑窗方式生成目标的检测框是一种非常常用的方法。而在实例分割任务中，比较主流的图像分割方法是首先检测目标边界框，然后进行裁剪和目标分割，如 Mask RCNN。在这篇工作中，我们研究了密集滑窗实例分割（dense sliding-window instance segmentation）的模式，发现与其他的密集预测任务如语义分割，目标检测不同，实例分割滑窗在每个空间位置的输出具有自己空间维度的几何结构。为了形式化这一点，我们提出了一个通用的框架 TensorMask 来获得这种几何结构。我们通过张量视图展示了相较于忽略这种结构的 baseline 方法，它可以有一个大的效果提升，甚至比肩于 Mask R-CNN。这样的实验结果足以说明TensorMask 为密集掩码预测任务提供了一个新的理解方向，并可以作为该领域新的基础方法。

## ThunderNet: Towards Real-time Generic Object Detection

https://blog.csdn.net/u011344545/article/details/88927579






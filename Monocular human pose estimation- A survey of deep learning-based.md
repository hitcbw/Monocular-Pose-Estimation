Monocular human pose estimation: A survey of deep learning-based methods  
---
Journal Type : Review papers from 2014 

---
Main Works: 
- categorize HPE method into four categories and introduce human body models. 
- Detailed 2D & 3D human pose estimation 
- Detailed some common datasets with evaluation protocols.
---
Abstract:    
Vision-based monocular human pose estimation, as one of the most fundamental and challenging problems
in computer vision, aims to obtain posture of the human body from input images or video sequences. The
recent developments of deep learning techniques have been brought significant progress and remarkable
breakthroughs in the field of human pose estimation. This survey extensively reviews the recent deep learningbased 2D and 3D human pose estimation methods published since 2014. This paper summarizes the challenges,
main frameworks, benchmark datasets, evaluation metrics, performance comparison, and discusses some
promising future research directions

---
Details:  
- [HPE method categories](#hpe-method-categories)
  - [Generative vs. Discriminative](#generative-vs-discriminative)
  - [Top-down vs. Bottom-up](#top-down-vs-bottom-up)
  - [Regression-based vs. Detection-based](#regression-based-vs-detection-based)
  - [One-stage vs. Multi-stage](#one-stage-vs-multi-stage)
- [Human Body Models](#human-body-models)
  - [Skeleton-based Model](#skeleton-based-model)
  - [Contour-based Model](#contour-based-model)
  - [Volume-based Model](#volume-based-model)
- [2D human pose estimation](#2d-human-pose-estimation)
  - [2D single person pose estimation](#2d-single-person-pose-estimation)
  - [Regression-based methods](#regression-based-methods)
  - [Detection-based methods](#detection-based-methods)
  - [2D multi-person pose estimation](#2d-multi-person-pose-estimation)
  - [Top-down methods](#top-down-methods)
  - [Bottom-up methods](#bottom-up-methods)
- [3D human pose estimation](#3d-human-pose-estimation)
  - [Model-free 3D single person pose estimation](#model-free-3d-single-person-pose-estimation)
  - [Model-based 3D single person pose estimation](#model-based-3d-single-person-pose-estimation)
  - [3D multi-person pose estimation](#3d-multi-person-pose-estimation)
- [Datasets and evaluation protocols](#datasets-and-evaluation-protocols)
  - [2D Datasets](#2d-datasets)
  - [2D Evaluation Metrics](#2d-evaluation-metrics)
  - [3D Datasets](#3d-datasets)
  - [## 3D Evaluation Metrics](#h2-id3d-evaluation-metrics-33d-evaluation-metricsh2)
# HPE method categories
## Generative vs. Discriminative
生成式方法与判别式方法的区别在于是否需要基于人体模型。生成式方法可以通过对模型结构的先验认知、从不同角度对二维或三维空间进行几何投影、回归方式对高维参数空间进行优化等方式进行处理；判别式方法可以基于学习直接建立源到姿态的映射或者在人体模板库中进行搜索。
## Top-down vs. Bottom-up
自顶向下指的是首先检测人体，在框出bounding box后对每个Box中的人体进行分析，这种方法的复杂度与人体数量成正相关。自底向上的方法先检测出所有的人体part，再通过人体模型适配或其他算法进行组装。这个part不仅包括人体关节，也可以是肢体（关节间的连接、模板组合[template patch]）
## Regression-based vs. Detection-based
## One-stage vs. Multi-stage
# Human Body Models
## Skeleton-based Model
## Contour-based Model
## Volume-based Model
# 2D human pose estimation
## 2D single person pose estimation
## Regression-based methods
## Detection-based methods
## 2D multi-person pose estimation
## Top-down methods
## Bottom-up methods
# 3D human pose estimation
## Model-free 3D single person pose estimation
## Model-based 3D single person pose estimation
## 3D multi-person pose estimation
# Datasets and evaluation protocols
## 2D Datasets
## 2D Evaluation Metrics
## 3D Datasets 
## 3D Evaluation Metrics
---

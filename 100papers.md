# CVPR 2020
## Title
[MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network](https://github.com/mkocabas/pose-residual-network)
## Abstract
In this paper, we present MultiPoseNet, a novel bottom-up multi-person pose estimation architecture that combines a multi-task model with a novel assignment method. MultiPoseNet can jointly handle person detection, person segmentation and pose estimation problems. The novel assignment method is implemented by the Pose Residual Network (PRN) which receives keypoint and person detections, and produces accurate poses by assigning keypoints to person instances. On the COCO keypoints dataset, our pose estimation method outperforms all previous bottom-up methods both in accuracy (+4-point mAP over previous best result) and speed; it also performs on par with the best top-down methods while being at least 4x faster. Our method is the fastest real time system with ∼ 23 frames/sec.
## Title
Domes to Drones: Self-Supervised Active
Triangulation for 3D Human Pose Reconstruction
## Abstract
Existing state-of-the-art estimation systems can detect 2d poses of multiple people in images quite reliably. In contrast, 3d pose estimation from a single image is ill-posed due to occlusion and depth ambiguities. Assuming access to multiple cameras, or given an active system able to position itself to observe the scene from multiple viewpoints, reconstructing 3d pose from 2d measurements becomes well-posed within the framework of standard multi-view geometry. Less clear is what is an informative set of viewpoints for accurate 3d reconstruction, particularly in complex scenes, where people are occluded by others or by scene objects. In order to address the view selection problem in a principled way, we here introduce ACTOR, an active triangulation agent for 3d human pose reconstruction. Our fully trainable agent consists of a 2d pose estimation network (any of which would work) and a deep reinforcement learning-based policy for camera viewpoint selection. The policy predicts observation viewpoints, the number of which varies adaptively depending on scene content, and the associated images are fed to an underlying pose estimator. Importantly, training the policy requires no annotations-given a 2d pose estimator, ACTOR is trained in a self-supervised manner. In extensive evaluations on complex multi-people scenes filmed in a Panoptic dome, under multiple viewpoints, we compare our active triangulation agent to strong multi-view baselines, and show that ACTOR produces significantly more accurate 3d pose reconstructions. We also provide a proof-of-concept experiment indicating the potential of connecting our view selection policy to a physical drone observer.

## Title
Cascaded Pyramid Network for Multi-Person Pose Estimation
Yilun
## Abstract
The topic of multi-person pose estimation has been largely improved recently, especially with the development of convolutional neural network. However, there still exist a lot of challenging cases, such as occluded keypoints, invisible keypoints and complex background, which cannot be well addressed. In this paper, we present a novel network structure called Cascaded Pyramid Network (CPN) which targets to relieve the problem from these 'hard' keypoints. More specifically, our algorithm includes two stages: GlobalNet and RefineNet. GlobalNet is a feature pyramid network which can successfully localize the 'simple' keypoints like eyes and hands but may fail to precisely recognize the occluded or invisible keypoints. Our RefineNet tries explicitly handling the 'hard' keypoints by integrating all levels of feature representations from the GlobalNet together with an online hard keypoint mining loss. In general, to address the multi-person pose estimation problem, a top-down pipeline is adopted to first generate a set of human bounding boxes based on a detector, followed by our CPN for keypoint localization in each human bounding box. Based on the proposed algorithm, we achieve state-of-art results on the COCO keypoint benchmark, with average precision at 73.0 on the COCO test-dev dataset and 72.1 on the COCO test-challenge dataset, which is a 19% relative improvement compared with 60.5 from the COCO 2016 keypoint challenge. Code1 and the detection results for person used will be publicly available for further research.
## Title
Multi-Scale Structure-Aware Network for Human Pose Estimation
## Abstract
We develop a robust multi-scale structure-aware neural network for human pose estimation. This method improves the recent deep conv-deconv hourglass models with four key improvements: (1) multi-scale supervision to strengthen contextual feature learning in matching body keypoints by combining feature heatmaps across scales, (2) multi-scale regression network at the end to globally optimize the structural matching of the multi-scale features, (3) structure-aware loss used in the intermediate supervision and at the regression to improve the matching of keypoints and respective neighbors to infer a higher-order matching configurations, and (4) a keypoint masking training scheme that can effectively fine-tune our network to robustly localize occluded keypoints via adjacent matches. Our method can effectively improve state-of-the-art pose estimation methods that suffer from difficulties in scale varieties, occlusions, and complex multi-person scenarios. This multi-scale supervision tightly integrates with the regression network to effectively (i) localize keypoints using the ensemble of multi-scale features, and (ii) infer global pose configuration by maximizing structural consistencies across multiple keypoints and scales. The keypoint masking training enhances these advantages to focus learning on hard occlusion samples. Our method achieves the leading position in the MPII challenge leaderboard among the state-of-the-art methods.
## Title
Efficient Online Multi-Person 2D Pose Tracking with Recurrent Spatio-Temporal Affinity Fields
## Abstract
We present an online approach to efficiently and simultaneously detect and track 2D poses of multiple people in a video sequence. We build upon Part Affinity Field (PAF) representation designed for static images, and propose an architecture that can encode and predict Spatio-Temporal Affinity Fields (STAF) across a video sequence. In particular, we propose a novel temporal topology cross-linked across limbs which can consistently handle body motions of a wide range of magnitudes. Additionally, we make the overall approach recurrent in nature, where the network ingests STAF heatmaps from previous frames and estimates those for the current frame. Our approach uses only online inference and tracking, and is currently the fastest and the most accurate bottom-up approach that is runtime-invariant to the number of people in the scene and accuracy-invariant to input frame rate of camera. Running at sim30 fps on a single GPU at single scale, it achieves highly competitive results on the PoseTrack benchmarks.
## Title
Monocular 3D Pose and Shape Estimation of Multiple People in Natural Scenes: The Importance of Multiple Scene Constraints
## Abstract
Human sensing has greatly benefited from recent advances in deep learning, parametric human modeling, and large scale 2d and 3d datasets. However, existing 3d models make strong assumptions about the scene, considering either a single person per image, full views of the person, a simple background or many cameras. In this paper, we leverage state-of-the-art deep multi-task neural networks and parametric human and scene modeling, towards a fully automatic monocular visual sensing system for multiple interacting people, which (i) infers the 2d and 3d pose and shape of multiple people from a single image, relying on detailed semantic representations at both model and image level, to guide a combined optimization with feedforward and feedback components, (ii) automatically integrates scene constraints including ground plane support and simultaneous volume occupancy by multiple people, and (iii) extends the single image model to video by optimally solving the temporal person assignment problem and imposing coherent temporal pose and motion reconstructions while preserving image alignment fidelity. We perform experiments on both single and multi-person datasets, and systematically evaluate each component of the model, showing improved performance and extensive multiple human sensing capability. We also apply our method to images with multiple people, severe occlusions and diverse backgrounds captured in challenging natural scenes, and obtain results of good perceptual quality.
## Title
Simple baselines for human pose estimation and tracking
## Abstract
There has been significant progress on pose estimation and increasing interests on pose tracking in recent years. At the same time, the overall algorithm and system complexity increases as well, making the algorithm analysis and comparison more difficult. This work provides simple and effective baseline methods. They are helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. The code will be available at https://github.com/leoxiaobin/pose.pytorch.
## Title
PoseTrack: A Benchmark for Human Pose Estimation and Tracking
## Abstract
Existing systems for video-based pose estimation and tracking struggle to perform well on realistic videos with multiple people and often fail to output body-pose trajectories consistent over time. To address this shortcoming this paper introduces PoseTrack which is a new large-scale benchmark for video-based human pose estimation and articulated tracking. Our new benchmark encompasses three tasks focusing on i) single-frame multi-person pose estimation, ii) multi-person pose estimation in videos, and iii) multi-person articulated tracking. To establish the benchmark, we collect, annotate and release a new dataset that features videos with multiple people labeled with person tracks and articulated pose. A public centralized evaluation server is provided to allow the research community to evaluate on a held-out test set. Furthermore, we conduct an extensive experimental study on recent approaches to articulated pose tracking and provide analysis of the strengths and weaknesses of the state of the art. We envision that the proposed benchmark will stimulate productive research both by providing a large and representative training dataset as well as providing a platform to objectively evaluate and compare the proposed methods. The benchmark is freely accessible at https://posetrack.net/.
## Title
End-to-end learning for graph decomposition
## Abstract
Deep neural networks provide powerful tools for pattern recognition, while classical graph algorithms are widely used to solve combinatorial problems. In computer vision, many tasks combine elements of both pattern recognition and graph reasoning. In this paper, we study how to connect deep networks with graph decomposition into an end-to-end trainable framework. More specifically, the minimum cost multicut problem is first converted to an unconstrained binary cubic formulation where cycle consistency constraints are incorporated into the objective function. The new optimization problem can be viewed as a Conditional Random Field (CRF) in which the random variables are associated with the binary edge labels. Cycle constraints are introduced into the CRF as high-order potentials. A standard Convolutional Neural Network (CNN) provides the front-end features for the fully differentiable CRF. The parameters of both parts are optimized in an end-to-end manner. The efficacy of the proposed learning algorithm is demonstrated via experiments on clustering MNIST images and on the challenging task of real-world multi-people pose estimation.
## Title
Ego-pose estimation and forecasting as real-time PD control
## Abstract
We propose the use of a proportional-derivative (PD) control based policy learned via reinforcement learning (RL) to estimate and forecast 3D human pose from egocentric videos. The method learns directly from unsegmented egocentric videos and motion capture data consisting of various complex human motions (e.g., crouching, hopping, bending, and motion transitions). We propose a video-conditioned recurrent control technique to forecast physically-valid and stable future motions of arbitrary length. We also introduce a value function based fail-safe mechanism which enables our method to run as a single pass algorithm over the video data. Experiments with both controlled and in-the-wild data show that our approach outperforms previous art in both quantitative metrics and visual quality of the motions, and is also robust enough to transfer directly to real-world scenarios. Additionally, our time analysis shows that the combined use of our pose estimation and forecasting can run at 30 FPS, making it suitable for real-time applications.
## Title
A cascaded inception of inception network with attention modulated feature fusion for human pose estimation
## Abstract
Accurate keypoint localization of human pose needs diversified features: the high level for contextual dependencies and the low level for detailed refinement of joints. However, the importance of the two factors varies from case to case, but how to efficiently use the features is still an open problem. Existing methods have limitations in preserving low level features, adaptively adjusting the importance of different levels of features, and modeling the human perception process. This paper presents three novel techniques step by step to efficiently utilize different levels of features for human pose estimation. Firstly, an inception of inception (IOI) block is designed to emphasize the low level features. Secondly, an attention mechanism is proposed to adjust the importance of individual levels according to the context. Thirdly, a cascaded network is proposed to sequentially localize the joints to enforce message passing from joints of stand-alone parts like head and torso to remote joints like wrist or ankle. Experimental results demonstrate that the proposed method achieves the state-of-the-art performance on both MPII and LSP benchmarks.
## Title
Cross-view person identification by matching human poses estimated with confidence on each body joint
## Abstract
Cross-view person identification (CVPI) from multiple temporally synchronized videos taken by multiple wearable cameras from different, varying views is a very challenging but important problem, which has attracted more interests recently. Current state-of-the-art performance of CVPI is achieved by matching appearance and motion features across videos, while the matching of pose features does not work effectively given the high inaccuracy of the 3D human pose estimation on videos/images collected in the wild. In this paper, we introduce a new metric of confidence to the 3D human pose estimation and show that the combination of the inaccurately estimated human pose and the inferred confidence metric can be used to boost the CVPI performance -the estimated pose information can be integrated to the appearance and motion features to achieve the new state-of-the-art CVPI performance. More specifically, the estimated confidence metric is measured at each human-body joint and the joints with higher confidence are weighted more in the pose matching for CVPI. In the experiments, we validate the proposed method on three wearable-camera video datasets and compare the performance against several other existing CVPI methods.
## Title
Sim2real transfer learning for 3D human pose estimation: motion to the rescue
## Abstract
Synthetic visual data can provide practically infinite diversity and rich labels, while avoiding ethical issues with privacy and bias. However, for many tasks, current models trained on synthetic data generalize poorly to real data. The task of 3D human pose estimation is a particularly interesting example of this sim2real problem, because learning-based approaches perform reasonably well given real training data, yet labeled 3D poses are extremely difficult to obtain in the wild, limiting scalability. In this paper, we show that standard neural-network approaches, which perform poorly when trained on synthetic RGB images, can perform well when the data is pre-processed to extract cues about the person's motion, notably as optical flow and the motion of 2D keypoints. Therefore, our results suggest that motion can be a simple way to bridge a sim2real gap when video is available. We evaluate on the 3D Poses in the Wild dataset, the most challenging modern benchmark for 3D pose estimation, where we show full 3D mesh recovery that is on par with state-of-the-art methods trained on real 3D sequences, despite training only on synthetic humans from the SURREAL dataset.
## Title
Chirality Nets for Human Pose Regression
## Abstract
We propose Chirality Nets, a family of deep nets that is equivariant to the "chirality transform," i.e., the transformation to create a chiral pair. Through parameter sharing, odd and even symmetry, we propose and prove variants of standard building blocks of deep nets that satisfy the equivariance property, including fully connected layers, convolutional layers, batch-normalization, and LSTM/GRU cells. The proposed layers lead to a more data efficient representation and a reduction in computation by exploiting symmetry. We evaluate chirality nets on the task of human pose regression, which naturally exploits the left/right mirroring of the human body. We study three pose regression tasks: 3D pose estimation from video, 2D pose forecasting, and skeleton based activity recognition. Our approach achieves/matches state-of-the-art results, with more significant gains on small datasets and limited-data settings.
## Title
On boosting single-frame 3D human pose estimation via monocular videos
## Abstract
The premise of training an accurate 3D human pose estimation network is the possession of huge amount of richly annotated training data. Nonetheless, manually obtaining rich and accurate annotations is, even not impossible, tedious and slow. In this paper, we propose to exploit monocular videos to complement the training dataset for the single-image 3D human pose estimation tasks. At the beginning, a baseline model is trained with a small set of annotations. By fixing some reliable estimations produced by the resulting model, our method automatically collects the annotations across the entire video as solving the 3D trajectory completion problem. Then, the baseline model is further trained with the collected annotations to learn the new poses. We evaluate our method on the broadly-adopted Human3.6M and MPI-INF-3DHP datasets. As illustrated in experiments, given only a small set of annotations, our method successfully makes the model to learn new poses from unlabelled monocular videos, promoting the accuracies of the baseline model by about 10%. By contrast with previous approaches, our method does not rely on either multi-view imagery or any explicit 2D keypoint annotations.
## Title
Resolving 3D human pose ambiguities with 3D scene constraints
## Abstract
To understand and analyze human behavior, we need to capture humans moving in, and interacting with, the world. Most existing methods perform 3D human pose estimation without explicitly considering the scene. We observe however that the world constrains the body and vice-versa. To motivate this, we show that current 3D human pose estimation methods produce results that are not consistent with the 3D scene. Our key contribution is to exploit static 3D scene structure to better estimate human pose from monocular images. The method enforces Proximal Relationships with Object eXclusion and is called PROX. To test this, we collect a new dataset composed of 12 different 3D scenes and RGB sequences of 20 subjects moving in and interacting with the scenes. We represent human pose using the 3D human body model SMPL-X and extend SMPLify-X to estimate body pose using scene constraints. We make use of the 3D scene information by formulating two main constraints. The inter-penetration constraint penalizes intersection between the body model and the surrounding 3D scene. The contact constraint encourages specific parts of the body to be in contact with scene surfaces if they are close enough in distance and orientation. For quantitative evaluation we capture a separate dataset with 180 RGB frames in which the ground-truth body pose is estimated using a motion capture system. We show quantitatively that introducing scene constraints significantly reduces 3D joint error and vertex error. Our code and data are available for research at https://prox.is.tue.mpg.de.
## Title
Resolving 3D human pose ambiguities with 3D scene constraints
## Abstract
To understand and analyze human behavior, we need to capture humans moving in, and interacting with, the world. Most existing methods perform 3D human pose estimation without explicitly considering the scene. We observe however that the world constrains the body and vice-versa. To motivate this, we show that current 3D human pose estimation methods produce results that are not consistent with the 3D scene. Our key contribution is to exploit static 3D scene structure to better estimate human pose from monocular images. The method enforces Proximal Relationships with Object eXclusion and is called PROX. To test this, we collect a new dataset composed of 12 different 3D scenes and RGB sequences of 20 subjects moving in and interacting with the scenes. We represent human pose using the 3D human body model SMPL-X and extend SMPLify-X to estimate body pose using scene constraints. We make use of the 3D scene information by formulating two main constraints. The inter-penetration constraint penalizes intersection between the body model and the surrounding 3D scene. The contact constraint encourages specific parts of the body to be in contact with scene surfaces if they are close enough in distance and orientation. For quantitative evaluation we capture a separate dataset with 180 RGB frames in which the ground-truth body pose is estimated using a motion capture system. We show quantitatively that introducing scene constraints significantly reduces 3D joint error and vertex error. Our code and data are available for research at https://prox.is.tue.mpg.de.
## Title
Camera distance-aware top-down approach for 3D multi-person pose estimation from a single RGB image
## Abstract
Although significant improvement has been achieved recently in 3D human pose estimation, most of the previous methods only treat a single-person case. In this work, we firstly propose a fully learning-based, camera distance-aware top-down approach for 3D multi-person pose estimation from a single RGB image. The pipeline of the proposed system consists of human detection, absolute 3D human root localization, and root-relative 3D single-person pose estimation modules. Our system achieves comparable results with the state-of-the-art 3D single-person pose estimation models without any groundtruth information and significantly outperforms previous 3D multi-person pose estimation methods on publicly available datasets. The code is available in footnote{url{https://github.com/mks0601/3DMPPE-ROOTNET-RELEASE}}textsuperscript{,}footnote{url{https://github.com/mks0601/3DMPPE-POSENET-RELEASE}}.
## Title
Polarimetric relative pose estimation
## Abstract
In this paper we consider the problem of relative pose estimation from two images with per-pixel polarimetric information. Using these additional measurements we derive a simple minimal solver for the essential matrix which only requires two point correspondences. The polarization constraints allow us to pointwise recover the 3D surface normal up to a two-fold ambiguity for the diffuse reflection. Since this ambiguity exists per point, there is a combinatorial explosion of possibilities. However, since our solver only requires two point correspondences, we only need to consider 16 configurations when solving for the relative pose. Once the relative orientation is recovered, we show that it is trivial to resolve the ambiguity for the remaining points. For robustness, we also propose a joint optimization between the relative pose and the refractive index to handle the refractive distortion. In experiments, on both synthetic and real data, we demonstrate that by leveraging the additional information available from polarization cameras, we can improve over classical methods which only rely on the 2D-point locations to estimate the geometry. Finally, we demonstrate the practical applicability of our approach by integrating it into a state-of-the-art global Structure-from-Motion pipeline.
## Title
HEMlets pose: Learning part-centric heatmap triplets for accurate 3D human pose estimation
## Abstract
Estimating 3D human pose from a single image is a challenging task. This work attempts to address the uncertainty of lifting the detected 2D joints to the 3D space by introducing an intermediate state - Part-Centric Heatmap Triplets (HEMlets), which shortens the gap between the 2D observation and the 3D interpretation. The HEMlets utilize three joint-heatmaps to represent the relative depth information of the end-joints for each skeletal body part. In our approach, a Convolutional Network(ConvNet) is first trained to predict HEMlests from the input image, followed by a volumetric joint-heatmap regression. We leverage on the integral operation to extract the joint locations from the volumetric heatmaps, guaranteeing end-to-end learning. Despite the simplicity of the network design, the quantitative comparisons show a significant performance improvement over the best-of-grade method (by 20% on Human3.6M). The proposed method naturally supports training with 'in-the-wild'' images, where only weakly-annotated relative depth information of skeletal joints is available. This further improves the generalization ability of our model, as validated by qualitative comparisons on outdoor images.
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract
## Title
## Abstract

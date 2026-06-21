# 1 2025 Spatial Intelligence Survey

- [Reconstructing 4D Spatial Intelligence: A Survey](https://arxiv.org/pdf/2507.21045)

- To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence

## Abstract

- Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. 
    - These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, 
    - to embodied AI, which emphasizes interaction modeling and physical realism. 
    - Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. 
    - Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. 
    
- To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence:
    - Level 1 – reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); 
    - Level 2 – reconstruction of 3D scene components (e.g., objects, humans, structures); 
    - Level 3 – reconstruction of 4D dynamic scenes; 
    - Level 4 – modeling of interactions among scene components; and 
    - Level 5 – incorporation of physical laws and constraints. 
    
- We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. 

## 1 INTRODUCTION

- THE automatic reconstruction of 4D spatial intelligence using machine learning or deep learning techniques has long been a crucial and challenging problem in computer vision. By capturing both the static configurations and dynamic changes over time, 4D spatial intelligence shall provide a comprehensive representation and understanding of the spatial environments that integrate the three dimensional geometric structures with their temporal evolution. 
    - This field has attracted significant attention due to its wide range of applications in video games, movies, and immersive experiences (AR/VR), where high fidelity 4D scenes serve as the foundation for delivering realistic user experiences. 
    - Beyond these applications that primarily focus on the fundamental components of 4D spatial intelligence – namely low-level cues such as depth, camera pose, point map, and 3D tracking, as well as scene composing elements and dynamics – spatial intelligence also plays a pivotal role in advancing **embodied AI and world models**. 
    - These latter domains place a strong emphasis on the interactions among scene components and the physical plausibility of the reconstructed environments.

- In recent years, techniques for reconstructing 4D spatial intelligence have seen rapid advancements. Several surveys have provided valuable perspectives from various angles and have highlighted persistent challenges in the field. 
    - For example, [11], [12], [13] reviewed the recent process in deep stereo matching to obtain the low-level scene information; 
    - [14], [15], [16] offered a comprehensive overview of advances in 3D scene reconstruction, covering a range of input modalities and diverse 3D representations;
    - [9], [10] classified dynamic 4D scene reconstruction methods into categories based on their core architectural principles.

- However, the field has advanced considerably, driven by the emergence of novel 3D representations [17], [18], [19], high-quality video generation techniques [20], [21], [22] that provide richer input data, and more efficient models capable of delivering superior reconstruction quality. 
    - Despite these strides, additionally, none of the existing surveys thoroughly examines the different compositional levels of the dynamic 4D scenes, nor do they offer a detailed analysis of their respective developments and open challenges. 
    - This would potentially lead to a fragmented understanding that overlooks critical components. 
    - These gaps highlight the need for a comprehensive, up-to-date survey that systematically categorizes 4D spatial intelligence into distinct levels, consolidates recent advancements, and maps the evolving landscape of 4D scene reconstruction.

- Driven by this urgent situation, we categorize the existing methods for reconstructing 4D spatial intelligence into five levels and provide a structured overview of their respective advances:
    - Level 1 – reconstruction of low-level 3D cues.
        - At Level 1, the system targets the reconstruction of fundamental 3D cues – namely, **depth, camera pose, point maps, and 3D tracking**. These low-level cues define the core structure of a 3D scene. 
        - Traditionally, this task has been broken down into separate subfields such as key point detection and matching, robust estimation, Structure-from-Motion (SfM), Bundle Adjustment (BA) and dense Multi-View Stereo (MVS). 
        - Recent approaches like DUSt3R and its followups aim to jointly solve these sub-problems, enabling more integrated and collaborative reasoning. 
        - **Building on transformer-based advances, [VGGT](https://arxiv.org/pdf/2503.11651) further introduces an end-to-end framework that rapidly estimates these low-level 3D cues within seconds.**
    - Level 2 – reconstruction of 3D scene components.
        - On top of level 1, level 2 methods move beyond basic 3D cues to reconstruct individual scene elements such as **humans, objects, and buildings.** 
        - While some methods may involve the composition or spatial arrangement of these elements within a scene, they generally do not model or enforce the correctness of their interactions. 
        - Recent methods for this level leverage the innovations in 3D representations like NeRF, 3D Gaussians, and meshes (DMTET, FlexiCube) to improve the reconstructed fine-scale details, rendering efficiency, and global structural coherence, **making the results ideal for photorealistic scene reconstruction and immersive virtual experiences.**
    - Level 3 – reconstruction of 4D dynamic scenes.
        - Level 3 incorporates dynamics into reconstructed 4D scenes, marking a key step toward enabling the “bullet time” experience of 4D spatial intelligence and delivering more immersive visual content. 
        - Existing approaches can generally be categorized into two main directions. 
            - The first line of work reconstructs a static canonical radiance field and models temporal changes through learned deformations over time. 
            - In contrast, the other type of techniques encode time directly as an additional parameter within the 3D representation, allowing for continuous modeling of scene dynamics.
    - Level 4 – modeling of interactions among scene components.
        - Advancing beyond the reconstruction of low-level cues, scene elements, and dynamics, Level 4 of spatial intelligence enters a more mature phase – focusing on modeling interactions between different components within a scene. 
        - Given that humans are often the central agents of interaction, early works primarily concentrated on capturing the motion of humans and manipulated objects observable in video inputs. 
        - With recent progress in 3D representations, recent methods have achieved more accurate reconstructions of both human and object appearances. 
        - Furthermore, the study of human-scene interactions has gained traction, serving as a foundational step toward constructing comprehensive **world models**.
    - Level 5 – Incorporation of physical laws and constraints.
        - While Level 4 systems are capable of modeling interactions between different scene components, they typically overlook underlying physical principles such as gravity, friction, and pressure. 
        - As a result, these methods may fall short in applications like embodied AI, where the goal is often to enable real-world robots to imitate actions and interactions observed in videos. Level 5 systems address this limitation by focusing on enforcing physical plausibility within reconstructed 4D scenes. 
        - Recent approaches leveraging platforms such as [IsaacGym (NVIDIA) and reinforcement learning techniques](https://arxiv.org/pdf/2108.10470), have demonstrated the ability to learn and replicate human-like skills directly from video data, marking a significant step toward physically grounded spatial intelligence. 
        - Beyond human-related applications, the physical modeling of general 3D objects, such as simulating object deformation, collisions, and dynamics, as well as physical scenes, has also become an active area of research, expanding the scope and applicability of Level 5 reconstruction systems.

- Scope. 
    - This survey primarily focuses on approaches for reconstructing 4D scenes from video inputs. 
    - Specifically, we examine key developments and representative works across our defined Levels 1 through 5 of the 4D spatial intelligence. 
    - The papers reviewed are predominantly drawn from leading conferences and journals in computer vision and computer graphics, along with select preprints released on arXiv in 2025. Our selection criteria emphasize relevance to the scope of this survey, with the goal of providing a comprehensive overview of recent rapid progress in the field.
    - We do not include the 3D generation methods and 4D generation approaches based on generative video diffusion models, as these methods typically yield a single type of input and have limited direct relevance to 4D reconstruction techniques. 
    - Additionally, this survey does not delve into a detailed analysis of various 3D representations. Readers interested in these complementary areas are encouraged to read existing surveys on 4D generation and the evolution of 3D representation methods

- Organization. 
    - An overview of the different levels of 4D spatial intelligence is illustrated in Fig. 1. 
    - In the following sections, we introduce a taxonomy that organizes recent research efforts according to the evolving process of reconstructing five key levels from video inputs: low-level 3D cues, basic 3D scene components, dynamic 4D scenes, interaction between scene components, and physics modeling. 
    - The overall structure of the survey is summarized in Fig. 1. Finally, in Sec. 7, we critically reflect on current methodologies, identify open challenges at each level of spatial intelligence, and discuss future directions for advancing 4D spatial intelligence beyond these five defined levels.

## 2 LEVEL 1 – LOW-LEVEL 3D CUES

- Depth, camera pose, and 3D tracking are commonly regarded as low-level cues in 3D scene modeling. 
    - These parameters capture the fundamental geometric and positional structure of the environment, forming the basis for higher level tasks such as object reconstruction, scene composition, and physical interaction modeling. In this sense, they function similarly to pixels and edges in 2D vision. 
    - As such, we define the reconstruction of these elements as level 1 of 4D spatial intelligence. 
    - The paradigms of the methods for obtaining these low-level cues from videos are illustrated in Fig. 3. They can be further categorized according to their respective objectives and the type of input videos.

### 2.1 Depth estimation

- Video-based depth estimation aims to generate accurate and temporally consistent depth maps from RGB video sequences. 

- Early approaches typically relied on inference time optimization to align depth across frames, or employed self-supervised warping using estimated egomotion and optical flow, often further enhanced by test-time refinement. 
    - While effective, these methods are computationally expensive and heavily dependent on the accuracy of pose and flow estimations. 
    - To address these challenges, feed-forward architectures have been introduced. 
        - Cost-volume–based methods construct 3D matching volumes to enforce temporal coherence, 
        - while flow-guided approaches integrate optical flow cues directly. 
        - Recurrent models leverage temporal recurrence to iteratively refine predictions across frames, 
        - and attention-based mechanisms dynamically re-weight temporal features.

- More recently, large-scale pretraining and diffusion-based frameworks have pushed the frontier further. 
    - DepthCrafter, ChronoDepth, and DepthAnyVideo **leverage video diffusion models to generate depth sequences directly**, 
    - **while [135] extends the ViT-based Depth Anything V2 for video depth estimation.** 
    - These models exhibit strong temporal consistency and robust generalization across diverse scenes. 

- **Overall, the field has progressed from optimization-heavy, pose-dependent pipelines to efficient feed-forward networks, and most recently, to pre-trained, diffusion-driven models that achieve both high accuracy and temporal coherence.**

### 2.2 Camera pose estimation

- Camera pose estimation from RGB videos can be generally solved by Visual Odometry (VO) algorithms, which are widely applied in robotics applications. 
    - **Classical geometry based VO methods** are typically categorized into two groups: feature-based and direct approaches. 
        - Feature-based VO estimates camera motion by detecting and tracking visual features across frames,
        - while direct VO infers motion by minimizing photometric error or applying feature warping. 
    - With the advent of deep learning, **learning-based VO methods have gained prominence, often outperforming traditional approaches in controlled settings but facing challenges in generalizing to unseen environments.** 
    - To overcome these limitations, **hybrid** methods have been proposed to combine learning-based techniques with geometric insights, leveraging the strengths of both paradigms.
    - More recently, to reduce reliance on manual hyperparameter tuning required by these hybrid methods, further studies have explored **reinforcement learning for adaptive decision-making in VO systems.** 

- It is also worth noting that **VO is closely related to Visual Simultaneous Localization and Mapping (VSLAM)**, which extends VO by concurrently constructing a map of the environment.
    - Methods that jointly estimate camera pose and dense depth for mapping purposes will be discussed in a later section on unified camera pose and depth estimation from video.

### 2.3 3D tracking

- 3D tracking estimation aims to recover the motion of scene elements in dynamic videos, providing temporally coherent correspondences in 3D space. 
    - A notable approach in this area is OmniMotion, which represents an input video using a quasi-3D canonical volume and performs dense, pixel-wise tracking by establishing bijective mappings between the local input space and the canonical space. Through per-video optimization, it jointly estimates the motion trajectories across the entire sequence, enabling consistent tracking over time. Building upon this framework, OmniTrackFast  enhances both computational efficiency and robustness by factorizing the underlying function representation into a local spatiotemporal feature grid, and further improves the model’s expressiveness by introducing non-linear functions into the coupling blocks. 
    - In contrast to these optimization-heavy methods, SpatialTracker proposes a feed-forward architecture that supports long range 3D tracking across videos without the need for test time optimization, offering a more scalable and efficient alternative. 
        - SceneTracker employs an iterative strategy to approximate the optimal 3D trajectory, dynamically indexing and constructing both appearance correlation and depth residual features in parallel. 
    - DELTA [163] introduces a coarse-to-fine trajectory estimation strategy, allowing for efficient dense tracking across the entire frame rather than being limited to a sparse set of locations. 
    - Seurat derives depth directly from 2D tracking inputs to recover 3D trajectories. 
    - TAPIP3D constructs spatio-temporal feature clouds from videos by utilizing depth and camera motion information to project 2D video features into a 3D world space, where the effects of camera movement are effectively neutralized. 
    - Recent methods, such as EgoPoints, introduces a new benchmark and new metrics for point tracking from egocentric videos. It opens the door for future works.

- Together, these methods illustrate the evolving landscape of 3D tracking, spanning from optimization-based pipelines to fully end-to-end learning systems.


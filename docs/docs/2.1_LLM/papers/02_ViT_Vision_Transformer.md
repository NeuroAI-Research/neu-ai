
# 2 2020 ViT: Vision Transformer

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

- Fine-tuning code and pre-trained models are available at https://github.com/google-research/vision_transformer

## ABSTRACT

- While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. 

- In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. 

- When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

## 1 INTRODUCTION

- Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). 

- Thanks to Transformers’ computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020). 
    - With the models and datasets growing, there is still no sign of saturating performance.

- In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). 

- Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). 
    - The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. 

- Therefore, in large-scale image recognition, classic ResNet-like architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020).

- Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, **with the fewest possible modifications**. 
    - To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. 
    - Image patches are treated the same way as tokens (words) in an NLP application. 
    - We train the model on image classification in supervised fashion.

- When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. 
    - This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.
    - However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. 
    - When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, and 77.63% on the VTAB suite of 19 tasks.

## 2 RELATED WORK

- Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks. 

- Large Transformer-based models are often pre-trained on large corpora and then fine-tuned for the task at hand: BERT (Devlin et al., 2019) uses a denoising self-supervised pre-training task, while the **GPT** line of work uses language modeling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).

- Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past. 
    - Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020). 
    - In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global self attention in order to be applicable to images. 
    - An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a). 

- Many of these specialized attention architectures demonstrate
promising results on computer vision tasks, but require complex engineering to be implemented efficiently on hardware accelerators.

- Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2 from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs. 
    - Moreover, Cordonnier et al. (2020) use a small patch size of 2 × 2 pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.

- There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms of self-attention, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen
et al., 2020c; Lu et al., 2019; Li et al., 2019).

- Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers to image pixels after reducing image resolution and color space. The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or probed linearly for classification performance, achieving a maximal accuracy of 72% on ImageNet.

- Our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. The use of additional data sources allows to achieve state-of-the-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020).

- Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from large scale datasets such as ImageNet-21k and JFT-300M. 

- We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models used in prior works.

## 3 METHOD

- In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible.
    - An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.

### 3.1 VISION TRANSFORMER (VIT)

![](../imgs/02_Vision_Transformer_ViT.png)

$$\begin{align}
& z_0 = [x_\text{class}; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos}  
&& E \in R^{P^2 C \times D}, E_{pos} \in R^{(N+1) \times D} \\
& z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1} && l = 1...L \\
& z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l && l = 1...L \\
& y = \text{LN}(z_L^0) &&
\end{align}$$

- An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. 

- To handle 2D images, we reshape the image $x \in R^{H \times W \times C}$ into a sequence of flattened 2D patches 
$x_p \in R^{N \times P^2 C}$, where $(H, W)$ is the resolution of the original image, $C$ is the number of channels, $(P, P)$ is the resolution of each image patch, and $N = H W / P^2$ is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. 

- The Transformer uses constant latent vector size $D$ through all of its layers, so we flatten the patches and map to $D$ dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the **patch embeddings**.

- Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches $(z_0^0 = x_\text{class})$, whose state at the output of the Transformer encoder $(z_L^0)$ serves as the
image representation $y$ (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to $z_L^0$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

- Position embeddings are added to the patch embeddings to retain positional information. **We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4).** The resulting sequence of embedding vectors serves as input to the encoder.

- The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). 
    - Layernorm (LN) is applied before every block, 
    - and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
    - The MLP contains two layers with a **GELU** non-linearity.

- Inductive bias. 
    - We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. 
    - In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. 
    - The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). 
    - Other than that, **the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.**

- Hybrid Architecture. 
    - As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). 
    - In this hybrid model, the patch embedding projection $E$ (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size $1 \times 1$, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension.
    - The classification input embedding and position embeddings are added as described above.

### 3.2 FINE-TUNING AND HIGHER RESOLUTION

- Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. 
    - For this, we remove the pre-trained prediction head and attach a zero-initialized $D \times K$ feedforward layer, where $K$ is the number of downstream classes. It is often beneficial to fine-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). 
    - When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. 
    - The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the **only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.**

## 4 EXPERIMENTS

- We evaluate the **representation learning** capabilities of ResNet, Vision Transformer (ViT), and the hybrid. 

- To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. 

- Lastly, we perform a small experiment using self-supervision, and show that **self-supervised ViT holds promise for the future**.

### 4.1 SETUP

### 4.2 COMPARISON TO STATE OF THE ART

### 4.3 PRE-TRAINING DATA REQUIREMENTS

### 4.4 SCALING STUDY

### 4.5 INSPECTING VISION TRANSFORMER

![](../imgs/02_inspecting_ViT_1.png)
![](../imgs/02_inspecting_ViT_2.png){width=200}

- To begin to understand how the Vision Transformer processes image data, we analyze its internal representations.

- The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional space (Eq. 1). 
    - Figure 7 (left) shows the top principal components of the the learned embedding filters. 
    - The components resemble plausible basis functions for a low-dimensional representation of the fine structure within each patch.

- After the projection, a learned position embedding is added to the patch representations. 
    - Figure 7 (center) shows that the model learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. 
    - Further, the row-column structure appears; patches in the same row/column have similar embeddings. 
    - Finally, a sinusoidal structure is sometimes apparent for larger grids (Appendix D). 
    - **That the position embeddings learn to represent 2D image topology explains why hand-crafted 2D-aware embedding variants do not yield improvements** (Appendix D.4).

- Self-attention allows ViT to integrate information across the entire image even in the lowest layers. 
    - We investigate to what degree the network makes use of this capability. Specifically, we compute the average distance in image space across which information is integrated, based on the attention weights (Figure 7, right). 
    - **This “attention distance” is analogous to receptive field size in CNNs.**
    - We find that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model. 
    - Other attention heads have consistently small attention distances in the low layers. This highly localized attention is less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right), **suggesting that it may serve a similar function as early convolutional layers in CNNs**. 
    - Further, the attention distance increases with network depth. Globally, we find that the model attends to image regions that are semantically relevant for classification (Figure 6).

### 4.6 SELF-SUPERVISION

- Transformers show impressive performance on NLP tasks. However, much of their success stems not only from their excellent scalability but also from large scale self-supervised pre-training (Devlin et al., 2019; Radford et al., 2018). 

- **We also perform a preliminary exploration on masked patch prediction for self-supervision, mimicking the masked language modeling task used in BERT**. 

- With self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training. Appendix B.1.2 contains further details. 

- We leave exploration of contrastive pre-training (Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Henaff et al., 2020) to future work. 

## 5 CONCLUSION

- We have explored the direct application of Transformers to image recognition. Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. 

- Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP. This **simple**,
yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets.

- Thus, Vision Transformer matches or exceeds the state of the art on many image classification datasets, whilst being relatively cheap to pre-train.

- While these initial results are encouraging, many challenges remain. 
    - One is to apply ViT to other computer vision tasks, such as detection and segmentation. Our results, coupled with those in Carion et al. (2020), indicate the promise of this approach. 
    - Another challenge is to continue exploring self-supervised pre-training methods. Our initial experiments show improvement from self-supervised pre-training, but there is still large gap between self-supervised and large-scale supervised pretraining. 
    - Finally, further scaling of ViT would likely lead to improved performance.


# 3 2021 CLIP: Contrastive Language Image Pretraining

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)

- We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP

## Abstract

- State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. 
    - This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. 

- Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.
    - We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. 
    - After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. 
    - We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification.
    - The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. 

## 1. Introduction and Motivating Work

- Pre-training methods which learn directly from raw text have revolutionized NLP over the last few years (Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Raffel et al., 2019).

- Task-agnostic objectives such as autoregressive and masked language modeling have scaled across many orders of magnitude in compute, model capacity, and data, steadily improving capabilities. 

- The development of “text-to-text” as a standardized input-output interface (McCann et al., 2018; Radford et al., 2019; Raffel et al., 2019) has enabled task agnostic architectures to zero-shot transfer to downstream datasets removing the need for specialized output heads or dataset specific customization. 

- Flagship systems like **GPT-3 (Brown et al., 2020)** are now competitive across many tasks with bespoke models while requiring little to no dataset specific training data.
    - These results suggest that the aggregate supervision accessible to modern pre-training methods within web-scale collections of text surpasses that of high-quality crowd-labeled NLP datasets. 

- However, in other fields such as computer vision it is still standard practice to pre-train models on crowd-labeled datasets such as ImageNet (Deng et al., 2009).

- Could scalable pre-training methods which learn directly from web text result in a similar breakthrough in computer vision? Prior work is encouraging.
    - Over 20 years ago Mori et al. (1999) explored improving content based image retrieval by training a model to predict the nouns and adjectives in text documents paired with images. 
    - Quattoni et al. (2007) demonstrated it was possible to learn more data efficient image representations via manifold learning in the weight space of classifiers trained to predict words in captions associated with images. 
    - Srivastava & Salakhutdinov (2012) explored **deep representation learning by training multimodal Deep Boltzmann Machines on top of low-level image and text tag features.**
    - Joulin et al. (2016) modernized this line of work and demonstrated that CNNs trained to predict words in image captions learn useful image representations. They converted the title, description, and hashtag metadata of images in the YFCC100M dataset (Thomee et al., 2016) into a bag-of-words multi-label classification task and showed that pretraining AlexNet (Krizhevsky et al., 2012) to predict these labels learned representations which preformed similarly to ImageNet-based pre-training on transfer tasks. 
    - Li et al. (2017) then extended this approach to predicting phrase ngrams in addition to individual words and demonstrated the ability of their system to zero-shot transfer to other image classification datasets by scoring target classes based on their dictionary of learned visual n-grams and predicting the one with the highest score. 

- Adopting more recent architectures and pre-training approaches, VirTex (Desai & Johnson, 2020), ICMLM (Bulent Sariyildiz et al., 2020), and ConVIRT (Zhang et al., 2020) have recently demonstrated the potential of transformer-based language modeling, masked language modeling, and contrastive objectives to learn image representations from text. 
    - While exciting as proofs of concept, using natural language supervision for image representation learning is still rare. 
    - This is likely because demonstrated performance on common benchmarks is much lower than alternative approaches.
    - For example, Li et al. (2017) reach only 11.5% accuracy on ImageNet in a zero-shot setting. This is well below the 88.4% accuracy of the current state of the art (Xie et al., 2020). It is even below the 50% accuracy of classic computer vision approaches (Deng et al., 2012). 

- Instead, more narrowly scoped but well-targeted uses of weak supervision have improved performance. 
    - Mahajan et al. (2018) showed that predicting ImageNet-related hashtags on Instagram images is an effective pre-training task. 
    - When fine-tuned to ImageNet these pre-trained models increased accuracy by over 5% and improved the overall state of the art at the time.
    - Kolesnikov et al. (2019) and Dosovitskiy et al. (2020) have also demonstrated large gains on a broader set of transfer benchmarks by pre-training models to predict the classes of the noisily labeled JFT-300M dataset.

- This line of work represents the current pragmatic middle ground between learning from a limited amount of supervised “gold-labels” and learning from practically unlimited amounts of raw text. 
    - However, it is not without compromises. Both works carefully design, and in the process limit, their supervision to 1000 and 18291 classes respectively.
    - Natural language is able to express, and therefore supervise, a much wider set of visual concepts through its generality. 
    - Both approaches also use static `softmax` classifiers to perform prediction and lack a mechanism for dynamic outputs. 
    - This severely curtails their flexibility and limits their  “zero-shot” capabilities.

- A crucial difference between these weakly supervised models and recent explorations of learning image representations directly from natural language is scale. 
    - While Mahajan et al. (2018) and Kolesnikov et al. (2019) trained their models for accelerator years on millions to billions of images, VirTex, ICMLM, and ConVIRT trained for accelerator days on one to two hundred thousand images. 
    
- In this work, we close this gap and study the behaviors of image classifiers trained with natural language supervision at large scale. Enabled by the large amounts of publicly available data of this form on the internet, we create a new dataset of **400 million (image, text) pairs** and demonstrate that a **simplified version of ConVIRT** trained from scratch, which we call CLIP, for Contrastive Language-Image Pre-training, is an efficient method of learning from natural language supervision. 
    - We study the scalability of CLIP by training a series of eight models spanning almost 2 orders of magnitude of compute and observe that transfer performance is a smoothly predictable function of compute (Hestness et al., 2017; Kaplan et al., 2020). 
    - We find that CLIP, similar to the GPT family, learns to perform a wide set of tasks during pre-training including OCR, geo-localization, action recognition, and many others.
    - We measure this by benchmarking the zero-shot transfer performance of CLIP on over 30 existing datasets and find it can be competitive with prior task-specific supervised models. 
    - We also confirm these findings with linear-probe representation learning analysis and show that CLIP outperforms the best publicly available ImageNet model while also being more computationally efficient. 
    - We additionally find that zero-shot CLIP models are much more robust than equivalent accuracy supervised ImageNet models which suggests that zero-shot evaluation of task-agnostic models is much more representative of a model’s capability. 
    - These results have significant policy and ethical implications, which we consider in Section 7.

## 2. Approach

### 2.1. Natural Language Supervision

- At the core of our approach is the idea of learning perception from supervision contained in natural language. 
    - As discussed in the introduction, this is not at all a new idea, however terminology used to describe work in this space is varied, even seemingly contradictory, and stated motivations are diverse. 
    - Zhang et al. (2020), Gomez et al. (2017), Joulin et al. (2016), and Desai & Johnson (2020) all introduce methods which **learn visual representations from text paired with images** but describe their approaches as unsupervised, self-supervised, weakly supervised, and supervised respectively.
    - **We emphasize that what is common across this line of work is not any of the details of the particular methods used but the appreciation of natural language as a training signal.** 
    - All these approaches are learning from natural language supervision. 
    - Although early work wrestled with the complexity of natural language when using topic model and n-gram representations, improvements in deep contextual representation learning suggest we now have the tools to effectively leverage this abundant source of supervision (McCann et al., 2017).

- Learning from natural language has several potential strengths over other training methods. 
    - It’s much easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. 
    - Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet. 
    - Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. 
    - In the following subsections, we detail the specific approach we settled on.

### 2.2. Creating a Sufficiently Large Dataset

- Existing work has mainly used three datasets, MS-COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017), and YFCC100M (Thomee et al., 2016). 
    - While MS-COCO and Visual Genome are high quality crowd-labeled datasets, they are small by modern standards with approximately 100,000 training photos each. By comparison, other computer vision systems are trained on up to 3.5 billion Instagram photos (Mahajan et al., 2018). YFCC100M, at 100 million photos, is a possible alternative, but the metadata for each image is sparse and of varying quality. Many images use automatically generated filenames like 20160716 113957.JPG as “titles” or contain “descriptions” of camera exposure settings. After filtering to keep only images with natural language titles and/or descriptions in English, the dataset shrunk by a factor of 6 to only 15 million photos. This is approximately the same size as ImageNet.

- A major motivation for natural language supervision is the large quantities of data of this form available publicly on the internet. Since existing datasets do not adequately reflect this possibility, considering results only on them would underestimate the potential of this line of research. 
- To address this, we constructed a new dataset of 400 million (image, text) pairs collected form a variety of publicly available sources on the Internet. 
    - To attempt to cover as broad a set of visual concepts as possible, we search for (image, text) pairs as part of the construction process whose text includes one of a set of 500,000 queries. 
    - We approximately class balance the results by including up to 20,000 (image, text) pairs per query. The resulting dataset has a similar total word count as the WebText dataset used to train GPT-2. We refer to this dataset as WIT for WebImageText.

### 2.3. Selecting an Efficient Pre-Training Method

- State-of-the-art computer vision systems use very large amounts of compute. Mahajan et al. (2018) required 19 GPU years to train their ResNeXt101-32x48d and Xie et al. (2020) required 33 TPUv3 core-years to train their Noisy Student EfficientNet-L2. 
    - When considering that both these systems were trained to predict only 1000 ImageNet classes, the task of learning an open set of visual concepts from natural language seems daunting. 
    
- In the course of our efforts, we found training efficiency was key to successfully scaling natural language supervision and we selected our final pre-training method based on this metric.

- **Our initial approach, similar to VirTex, jointly trained an image CNN and text transformer from scratch to predict the caption of an image.** 
    - However, we encountered difficulties efficiently scaling this method. In Figure 2 we show that a 63 million parameter transformer language model, which already uses twice the compute of its ResNet-50 image encoder, learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-of-words encoding of the same text.

- Both these approaches share a key similarity. They try to predict the exact words of the text accompanying each image.
    - This is a difficult task due to the wide variety of descriptions, comments, and related text that co-occur with images. 

- Recent work in **contrastive representation learning** for images has found that **contrastive objectives** can learn better representations than their **equivalent predictive objective** (Tian et al., 2019). 

- Other work has found that **although generative models of images can learn high quality image representations, they require over an order of magnitude more compute than contrastive models with the same performance** (Chen et al., 2020a). 

- Noting these findings, we explored training a system to solve the potentially easier **proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text.** 

![](../imgs/03_CLIP_vs_others.png){width=400}

- Starting with the same bag-of-words encoding baseline, we swapped the predictive objective for a contrastive objective in Figure 2 and observed a further 4x efficiency improvement in the rate of zero-shot transfer to ImageNet.

![](../imgs/03_CLIP.png)

- **Given a batch of $N$ (image, text) pairs, CLIP is trained to predict which of the $N \times N$ possible (image, text) pairings across a batch actually occurred.** 
    - To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the $N$ real pairs in the batch while minimizing the cosine similarity of the embeddings of the $N^2 − N$ incorrect pairings. 
    - We optimize a symmetric cross entropy loss over these similarity scores. 

```python
# Figure 3. Numpy-like pseudocode for the core of an implementation of CLIP.
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

- In Figure 3 we include pseudocode of the core of an implementation of CLIP. 
    - To our knowledge this batch construction technique and objective was first introduced in the area of deep metric learning as the **multi-class N-pair loss** Sohn (2016), was popularized for contrastive representation learning by Oord et al. (2018) as the **InfoNCE loss**, and was recently adapted for contrastive (text, image) representation learning in the domain of medical imaging by Zhang et al. (2020).

- Due to the large size of our pre-training dataset, over-fitting is not a major concern and the details of training CLIP are simplified compared to the implementation of Zhang et al. (2020). 

- We **train CLIP from scratch** without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights. 

- We do not use the non-linear projection between the representation and the contrastive embedding space, a change which was introduced by Bachman et al. (2019) and popularized by Chen et al. (2020b).
    - **We instead use only a linear projection to map from each encoder’s representation to the multi-modal embedding space.**
    - We did not notice a difference in training efficiency between the two versions and speculate that non-linear projections may be co-adapted with details of current image only in self-supervised representation learning methods. 
    - We also remove the text transformation function $t_u$ from Zhang et al. (2020) which samples a single sentence at uniform from the text since many of the (image, text) pairs in CLIP’s pretraining dataset are only a single sentence. 
    - We also simplify the image transformation function $t_v$. A random square crop from resized images is the only data augmentation used during training. 
    - Finally, the temperature parameter which controls the range of the logits in the softmax, $\tau$, is directly optimized during training as a log-parameterized multiplicative scalar to avoid turning as a hyper-parameter.

### 2.4. Choosing and Scaling a Model

- We consider two different architectures for the image encoder. 
    - For the first, we use **ResNet-50** (He et al., 2016a) as the base architecture for the image encoder due to its widespread adoption and proven performance. We make several modifications to the original version using the **ResNetD** improvements from He et al. (2019) and the antialiased rect-2 blur pooling from Zhang (2019). We also replace the global average pooling layer with an attention pooling mechanism. The attention pooling is implemented as a single layer of “transformer-style” multi-head QKV attention where the query is conditioned on the global average-pooled representation of the image. 
    - For the second architecture, we experiment with the recently introduced **Vision Transformer (ViT)** (Dosovitskiy et al., 2020). We closely follow their implementation with only the minor modification of adding an additional layer normalization to the combined patch and position embeddings before the transformer and use a slightly different initialization scheme.

- The text encoder is a **Transformer** (Vaswani et al., 2017) with the architecture modifications described in Radford et al. (2019, GPT-2). 
    - As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased **byte pair encoding (BPE)** representation of the text with a 49,152 vocab size (Sennrich et al., 2015). 
    - For computational efficiency, the max sequence length was capped at 76. 
    - The text sequence is bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space. 
    - Masked self-attention was used in the text encoder to preserve the ability to initialize with a pre-trained language model or add language modeling as an auxiliary objective, though exploration of this is left as future work.

- While previous computer vision research has often scaled models by increasing the width (Mahajan et al., 2018) or depth (He et al., 2016a) in isolation, for the ResNet image encoders we adapt the approach of Tan & Le (2019) which found that allocating additional compute across all of width, depth, and resolution outperforms only allocating it to only one dimension of the model. 
    - While Tan & Le (2019) tune the ratio of compute allocated to each dimension for their EfficientNet architecture, we use a simple baseline of allocating additional compute equally to increasing the width, depth, and resolution of the model. 
    - For the text encoder, we only scale the width of the model to be proportional to the calculated increase in width of the ResNet and do not scale the depth at all, as we found CLIP’s performance to be less sensitive to the capacity of the text encoder.

### 2.5. Training

- We train a series of 5 ResNets and 3 Vision Transformers.
    - For the ResNets we train a ResNet-50, a ResNet-101, and then 3 more which follow EfficientNet-style model scaling and use approximately 4x, 16x, and 64x the compute of a ResNet-50. They are denoted as RN50x4, RN50x16, and RN50x64 respectively. 
    - For the Vision Transformers we train a ViT-B/32, a ViT-B/16, and a ViT-L/14. 
    - We train all models for 32 epochs. 
    - We use the Adam optimizer (Kingma & Ba, 2014) with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all weights that are not gains or biases, and decay the learning rate using a cosine schedule (Loshchilov & Hutter, 2016). 
    - Initial hyperparameters were set using a combination of grid searches, random search, and manual tuning on the baseline ResNet50 model when trained for 1 epoch. Hyper-parameters were then adapted heuristically for larger models due to computational constraints. 
    - The learnable temperature parameter $\tau$ was initialized to the equivalent of 0.07 from (Wu et al., 2018) and clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability. 
    - We use a very large minibatch size of 32,768. 
    - Mixed-precision (Micikevicius et al., 2017) was used to accelerate training and save memory. To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016), half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded text encoder weights were used. 
    - The calculation of embedding similarities was also sharded with individual GPUs computing only the subset of the pairwise similarities necessary for their local batch of embeddings. 

- The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs. 

- For the ViT-L/14 we also pre-train at a higher 336 pixel resolution for one additional epoch to boost performance similar to FixRes (Touvron et al., 2019). We denote this model as **ViT-L/14@336px**. Unless otherwise specified, all results reported in this paper as “CLIP” use this model which we found to perform best.

## 3. Experiments

### 3.1. Zero-Shot Transfer

### 3.2. Representation Learning

### 3.3. Robustness to Natural Distribution Shift

## 4. Comparison to Human Performance

## 5. Data Overlap Analysis

## 6. Limitations

## 7. Broader Impacts

## 8. Related Work

## 9. Conclusion

- We have investigated whether it is possible to transfer the success of task-agnostic web-scale pre-training in NLP to another domain. 
- We find that adopting this formula results in similar behaviors emerging in the field of computer vision and discuss the social implications of this line of research. 
- In order to optimize their training objective, CLIP models learn to perform a wide variety of tasks during pretraining. This task learning can then be leveraged via natural language prompting to enable zero-shot transfer to many existing datasets. 
- At sufficient scale, the performance of this approach can be competitive with task-specific supervised models although there is still room for much improvement.

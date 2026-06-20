# 3 2023 DreamerV3

- [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104)
- [A public implementation of Dreamer that reproduces all results is available on the project website.](https://github.com/danijar/dreamerv3)

## Highlights

- Benchmark:
    - Crucially, Dreamer substantially outperforms `PPO` across all domains.
    - Dreamer outperforms the powerful `MuZero` algorithm while using only a fraction of the computational resources. 
    - Dreamer also outperforms the widely-used expert algorithms `Rainbow` and `IQN`
    - Dreamer outperforms the best remaining methods, including the `transformer-based IRIS` and `TWM` agents, the model-free `SPR`, and `SimPLe`

- Generations:
    - The DreamerV1 algorithm was limited to continuous control,
    - the DreamerV2 algorithm surpassed human performance on Atari,  
    - the DreamerV3 algorithm enables out-of-the-box learning across diverse benchmarks.

- We summarize the changes that DreamerV3 introduces as follows:
    - **Robustness techniques:** 
        - **Observation symlog,** 
        - **KL balance and free bits,** 
        - **1% unimix for all categoricals,** 
        - **percentile return normalization,** 
        - **symexp twohot loss** for the reward head and critic.
    - **Network architecture:** 
        - **Block GRU**, 
        - **RMSNorm normalization**, 
        - **SiLu activation**.
    - **Optimizer:** 
        - **Adaptive gradient clipping (AGC)**, 
        - **LaProp (RMSProp followed by momentum)**.
    - **Replay buffer**: 
        - **Larger capacity**, 
        - **online queue**, 
        - **storing and updating latent states.**

## Abstract

- Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence. 
    - Although current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, configuring them for new application domains requires significant human expertise and experimentation. 
    
- **We present DreamerV3, a general algorithm that outperforms specialized methods across over 150 diverse tasks, with a single configuration.** 

- **Dreamer learns a model of the environment and improves its behavior by imagining future scenarios.** 
    - Robustness techniques based on normalization, balancing, and transformations enable stable learning across domains. 
    - Applied out of the box, Dreamer is the **first algorithm** to collect diamonds in Minecraft from scratch without human data or curricula. 
    - This achievement has been posed as a significant challenge in artificial intelligence that requires exploring farsighted strategies from pixels and sparse rewards in an open world. 

- Our work allows solving challenging control problems without extensive experimentation, making reinforcement learning broadly applicable.

## Introduction

- Reinforcement learning has enabled computers to solve tasks through interaction, such as 
    - surpassing humans in the games of `Go and Dota`. 
    - It is also a key component for improving `large language models` beyond what is demonstrated in their pretraining data. 
    
- While **PPO** has become a standard algorithm in the field of reinforcement learning, more specialized algorithms are often employed to achieve higher performance. 
    - These specialized algorithms target the unique challenges posed by different application domains, such as 
        - continuous control 
        - discrete actions
        - sparse rewards
        - image inputs
        - spatial environments
        - board games
    - However, applying reinforcement learning algorithms to sufficiently new tasks — such as moving from video games to robotics tasks — requires substantial effort, expertise, and computational resources for tweaking the hyperparameters of the algorithm. 
    - This brittleness poses a bottleneck in applying reinforcement learning to new problems and also limits the applicability of reinforcement learning to computationally expensive models or tasks where tuning is prohibitive. 
    
- Creating a general algorithm that learns to master new domains without having to be reconfigured has been a central challenge in artificial intelligence and would open up reinforcement learning to a wide range of practical applications.

- **We present Dreamer, a general algorithm that outperforms specialized expert algorithms across a wide range of domains while using fixed hyperparameters, making reinforcement learning readily applicable to new problems.** 
    - The algorithm is based on the idea of learning a **world model** that equips the agent with rich perception and the ability to imagine the future. 
    - The `world model` predicts the outcomes of potential actions, 
    - a `critic neural network` judges the value of each outcome, 
    - and an `actor neural network` chooses actions to reach the best outcomes. 
    
- **Although intuitively appealing, robustly learning and leveraging world models to achieve strong task performance has been an open problem.** 
    - Dreamer overcomes this challenge through a range of robustness techniques based on 
        - normalization, 
        - balancing, 
        - transformations. 
    - We observe robust learning not only across over `150` tasks from the domains summarized in Figure 2, **but also across model sizes and training budgets**, offering a predictable way to increase performance. 
    - Notably, larger model sizes not only achieve higher scores but also require less interaction to solve a task.

![](../imgs/03_DreamerV3_domains.png)

- To push the boundaries of reinforcement learning, we consider the popular video game `Minecraft` that has become a focal point of research in recent years, with international competitions held for developing algorithms that autonomously learn to collect diamonds in Minecraft.
    - Solving this problem without human data has been widely recognized as a substantial challenge for artificial intelligence because of the 
        - sparse rewards, 
        - exploration difficulty, 
        - long time horizons, 
        - procedural diversity of this open world game. 
    - Due to these obstacles, previous approaches resorted to using **human expert data and domain-specific curricula**. 
    - **Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch.**

## Learning algorithm

- We present the third generation of the Dreamer algorithm. 
    - The algorithm consists of **three neural networks**: 
        - the `world model predicts` the outcomes of potential actions, 
        - the `critic judges` the value of each outcome, 
        - the `actor chooses` actions to reach the most valuable outcomes. 
    - The components are trained concurrently from `replayed experience` while the agent interacts with the environment. 
    - To succeed across domains, all three components need to accommodate **different signal magnitudes** and robustly balance terms in their objectives. 
        - This is challenging as we are not only targeting similar tasks within the same domain but aim to learn across diverse domains with fixed hyperparameters.
    - This section introduces the world model, critic, and actor along with their robust loss functions, as well as tools for robustly predicting quantities of unknown orders of magnitude.

### World model learning

- The world model learns **compact representations of sensory inputs** through `autoencoding` and enables `planning` by **predicting future representations and rewards for potential actions.** 

![](../imgs/03_DreamerV3_Learning.png)

- Figure 3: Training process of Dreamer. 
    - The world model encodes sensory inputs into **!!! discrete representations !!!** $z_t$ that are predicted by a sequence model with recurrent state $h_t$ given actions $a_t$
    - The inputs are reconstructed to shape the representations. 
    - The actor and critic predict actions $a_t$ and values $v_t$ and learn from trajectories of abstract representations predicted by the world model.

- We implement the world model as a [**Recurrent State-Space Model (RSSM)**](https://arxiv.org/pdf/1811.04551), shown in Figure 3. 
    - First, an `encoder` maps sensory inputs $x_t$ to **stochastic** representations $z_t$. 
    - Then, a sequence model with recurrent state $h_t$ predicts the sequence of these representations given past actions $a_{t−1}$. 
    - The concatenation of $h_t$ and $z_t$ forms the **model state** from which we predict rewards $r_t$ and episode continuation flags $c_t \in \{0, 1\}$ and reconstruct the inputs to ensure informative representations:

$$
\text{RSSM} \begin{cases}
\text{Sequence model:} & h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:} & z_t \sim q_\phi(z_t | h_t, x_t) \\
\text{Dynamics predictor:} & \hat{z}_t \sim p_\phi(\hat{z}_t | h_t) \\
\end{cases} \\
\begin{cases}
\text{Reward predictor:} & \hat{r}_t \sim p_\phi (\hat{r}_t | h_t, z_t) \\
\text{Continue predictor:} & \hat{c}_t \sim p_\phi(\hat{c}_t | h_t, z_t) \\
\text{Decoder:} & \hat{x}_t \sim p_\phi(\hat{x}_t | h_t, z_t)
\end{cases}
$$

- Figure 4 visualizes **long-term video predictions** of the world model. 
    - The encoder and decoder use 
        - `convolutional neural networks (CNN)` for image inputs 
        - `multi-layer perceptrons (MLPs)` for vector inputs 
    - The dynamics, reward, and continue predictors are also `MLPs`.
    
- **The representations are sampled from a vector of softmax distributions and we take straight-through gradients through the sampling step**

- Given a sequence batch of inputs $x_{1:T}$, actions $a_{1:T}$, rewards $r_{1:T}$, and continuation flags $c_{1:T}$, the world model parameters $\phi$ are optimized **end-to-end** to minimize the prediction loss $L_\text{pred}$, the dynamics loss $L_\text{dyn}$, and the representation loss $L_\text{rep}$ with corresponding loss weights $\beta_\text{pred} = 1, \beta_\text{dyn} = 1, \beta_\text{rep} = 0.1$:

$$ L(\phi) = E_{q_\phi}[ \sum_{t=1}^T \sum_k \beta_k L_k(\phi) ] $$

- The **prediction loss** trains 
    - the `decoder` and `reward predictor` via the **symlog squared loss** described later, 
    - the `continue predictor` via **logistic regression**. 
- The **dynamics loss** trains 
    - the `sequence model` to predict the next representation by minimizing the **KL divergence** between 
        - the predictor $p_\phi(z_t|h_t)$ and 
        - the next stochastic representation $q_\phi(z_t|h_t,x_t)$. 
- The **representation loss**, in turn, trains 
    - the representations to become more predictable allowing us to use a factorized dynamics predictor for fast sampling during imagination training. 

- The two losses differ in the **stop-gradient operator** $\text{sg}(\cdot)$ and their loss scale. 
- To avoid a degenerate solution where the dynamics are trivial to predict but fail to contain enough information about the inputs, we employ free bits by clipping the dynamics and representation losses below the value of $1 \text{nat} \approx 1.44 \text{bits}$. 
- This disables them while they are already minimized well to focus learning on the prediction loss:

$$
L_\text{pred}(\phi) = -\ln p_\phi(x_t|z_t, h_t) - \ln p_\phi(r_t|z_t, h_t) - \ln p_\phi(c_t|z_t, h_t) \\[5pt]
L_\text{dyn}(\phi) = \max(1, \text{KL}[ \; \text{sg}(q_\phi(z_t|h_t, x_t))  \; \parallel \; p_\phi(z_t|h_t) \; ]) \\[5pt]
L_\text{rep}(\phi) = \max(1, \text{KL}[ \; q_\phi(z_t|h_t, x_t)  \; \parallel \; \text{sg}( p_\phi(z_t|h_t) ) \; ]) 
$$

- **Study note: Negative Log-Likelihood ($-\ln p$) is Mean Squared Error (MSE)**
    - When the paper writes $-\ln p_\phi(x_t|z_t, h_t)$, it is treating the decoder as a probability distribution predictor.
    - The **standard assumption** for continuous data is that the decoder predicts the mean ($\mu$) of a normal distribution, with a fixed variance ($\sigma^2 = 1$).
    - The **probability density function** of a normal distribution for a prediction $\hat{x}$ given the truth $x$ is:
    
    $$ p(x) = {1 \over \sqrt{2\pi}} \exp\left( -{(x - \hat{x})^2 \over 2} \right) \\[5pt] 
    \Rightarrow -\ln p(x) \propto (x - \hat{x})^2 $$

    - **Minimizing MSE loss is mathematically identical to minimizing the NLL of a normal distribution.**

---

- Previous world models require scaling the representation loss differently based on the visual complexity of the environment. 
    - Complex 3D environments contain details unnecessary for control and thus prompt a stronger regularizer to simplify the representations and make them more predictable. 
    - In games with static backgrounds and where individual pixels may matter for the task, a weak regularizer is required to extract fine details. 
    - We find that combining free bits with a small representation loss resolves this dilemma, allowing for fixed hyperparameters across domains. 
    
- Moreover, we transform vector observations using the `symlog` function described later, to prevent large inputs and large reconstruction gradients, further stabilizing the trade-off with the representation loss.

- We occasionally observed spikes the in KL losses in earlier experiments, consistent with reports for **deep variational autoencoders**. 
    - To prevent this, we parameterize the categorical distributions of the encoder and dynamics predictor as mixtures of 1% uniform and 99% neural network output, making it impossible for them to become deterministic and thus ensuring well-behaved KL losses. 

- Further model details and hyperparameters are included in the supplementary material.

### Critic learning

- **The actor and critic neural networks learn behaviors purely from abstract trajectories of representations predicted by the world model**

- For environment interaction, we select actions by sampling from the actor network without lookahead planning. 

- **Markovian Model States**
    - The actor and critic do not look at `raw image` inputs. 
    - Instead, they operate on the compact model state:
    
    $$ s_t := \{h_t, z_t\} $$ 
    
    - $h_t$: The deterministic recurrent state (**memory of the past**)
    - $z_t$: The stochastic discrete representation (**representation of the current moment**)
    - Combining them ensures the state is Markovian, meaning $s_t$ **contains all the necessary information** to predict the future, making historical data redundant.

- The actor aims to maximize the return 

    $$ R_t := \sum_{\tau=0}^\infty \gamma^\tau r_{t+\tau} $$ 

    - with a discount factor $\gamma = 0.997$ for each model state. 
    
- To consider **rewards beyond the prediction horizon $T = 16$**, the critic learns to approximate the distribution of returns for each state under the current actor behavior:

    $$ \text{Actor: } a_t \sim \pi_\theta(a_t|s_t) \quad \text{Critic: } v_\psi(R_t|s_t) $$

    - Starting from representations of replayed inputs, the world model and actor generate a trajectory of 
        - **imagined model states** $s_{1:T}$, 
        - actions $a_{1:T}$, 
        - rewards $r_{1:T}$, 
        - continuation flags $c_{1:T}$ 
    - **Because the critic predicts a distribution, we read out its predicted values as the expectation of the distribution**:
        
        $$\boxed{ v_t := E[v_\psi(R_t|s_t)] } $$ 
    
    - To estimate returns that consider **rewards beyond the prediction horizon**, we compute bootstrapped $\lambda$-returns that integrate the predicted rewards and the values. 
    - The critic learns to predict the distribution of the return estimates $R_t^\lambda$ using the NLL loss:

    $$ L(\psi) := - \sum_{t=1}^T \ln p_\psi(R_t^\lambda | s_t) \\[5pt]
    R_t^\lambda := r_t + \gamma c_t \left( (1-\lambda) \boxed{\color{red} v_{t+1}} + \lambda R_{t+1}^\lambda \right) \quad R_T^\lambda := v_T $$

    - **Note: the paper uses $v_t$ instead of $v_{t+1}$ above, which I believe is a typo**
    - **While a simple choice would be to parameterize the critic as a Normal distribution, the return distribution can have multiple modes and vary by orders of magnitude across environments.** 
    - **To stabilize and accelerate learning under these conditions, we parameterize the critic as `categorical distribution with exponentially spaced bins`, decoupling the scale of gradients from the prediction targets as described later.** 
    - To improve value prediction in environments where rewards are challenging to predict, we apply the critic loss both 
        - to **imagined trajectories** with loss scale $\beta_\text{val} = 1$ 
        - and to trajectories sampled from the **replay buffer** with loss scale $\beta_\text{rep\_val} = 0.3$
    - The critic replay loss uses the imagination returns $R_t^\lambda$ at the start states of the imagination rollouts as on-policy value annotations for the replay trajectory to then compute $\lambda$-returns over the replay rewards.

- **Because the critic regresses targets that `depend on its own predictions`, we `stabilize learning` by regularizing the critic towards predicting the outputs of an `exponentially moving average of its own parameters`.** 
    - This is similar to target networks used previously in reinforcement learning but allows us to compute returns using the current critic network. 

- **We further noticed that the randomly initialized reward predictor and critic networks at the start of training can result in large predicted rewards that can delay the onset of learning.** 
    - **We thus initialize the output weight matrix of the reward predictor and critic to zeros, which alleviates the problem and accelerates early learning.**

### Actor learning

- The actor learns to choose actions that **maximize return** while exploring through an **entropy regularizer**. 
    - However, the correct scale for this regularizer depends both on the scale and frequency of rewards in the environment. 
    - Ideally, we would like the agent to explore more if rewards are sparse and exploit more if rewards are dense or nearby. 
    - At the same time, the exploration amount should not be influenced by arbitrary scaling of rewards in the environment. 
    - This requires normalizing the return scale while preserving information about reward frequency.
    - To use a **fixed entropy scale** of $\eta = 3 \times 10^{−4}$ across domains, **we normalize returns to be approximately contained in the interval $[0, 1]$**. 
    - **In practice, subtracting an offset from the returns does not change the actor gradient and thus dividing by the range $S$ is sufficient.** 
    - Moreover, to avoid amplifying noise from function approximation under sparse rewards, we only scale down large return magnitudes but leave small returns below the threshold of $L = 1$ untouched. 
    - We use the [**REINFORCE (1992) estimator**](https://link.springer.com/article/10.1007/BF00992696) for both discrete and continuous actions, resulting in the **surrogate (代理) loss function**:

$$ L(\theta) := -\sum_{t=1}^T A \log \pi_\theta(a_t|s_t) + \eta H[ \pi_\theta(a_t|s_t) ] \\[5pt]
A:= \text{sg}\left({ R_t^\lambda - v_\psi(s_t) \over \max(1, S) }\right) $$

- The return distribution can be multi-modal and include outliers, especially for randomized environments where some episodes have higher achievable returns than others. 
    - Normalizing by the smallest and largest observed returns would then scale returns down too much and may cause suboptimal convergence. 
    - To be robust to these outliers, we compute the range from the 5th to the 95th return percentile over the return batch and smooth out the estimate using an exponential moving average:

$$ S := \text{EMA}( \text{Per}(R_t^\lambda, 95) - \text{Per}(R_t^\lambda, 5), \; 0.99) $$

- **Previous work typically normalizes advantages rather than returns, which puts a fixed amount of emphasis on maximizing returns over entropy regardless of whether rewards are within reach.**
    - Scaling up advantages when rewards are sparse can amplify noise that outweighs the entropy regularizer and stagnates exploration. 
    - Normalizing rewards or returns by standard deviation can fail under sparse rewards where their standard deviation is near zero, drastically amplifying rewards regardless of their size. 
    - Constrained optimization targets a fixed entropy on average across states regardless of achievable returns, which is robust but explores slowly under sparse rewards and converges lower under dense rewards. 
    - **We did not find stable hyperparameters across domains for these approaches.** 
    
- **Return normalization with a denominator limit overcomes these challenges, exploring rapidly under sparse rewards and converging to high performance across diverse domains.**


### Robust predictions

- **Reconstructing inputs and predicting rewards and returns can be challenging because the scale of these quantities can vary across domains.** 
    - Predicting large targets using a `squared loss` can lead to divergence whereas `absolute` and `Huber losses` stagnate (停滞) learning. 
    - On the other hand, normalizing targets based on running statistics introduces non-stationarity into the optimization. 
    - **We suggest the `symlog squared error` as a simple solution to this dilemma.** 
    
- For this, a neural network $f_\theta(x)$ with inputs $x$ and parameters $\theta$ learns to **predict a transformed version of its targets $y$**. 
    - To read out predictions $\hat{y}$ of the network, we apply the inverse transformation:
    - **Using the logarithm as transformation would not allow us to predict targets that take on negative values. Therefore, we choose a function from the bi-symmetric logarithmic family that we name `symlog` as the transformation with the `symexp` function as its inverse:**

$$ L(\theta) := {1\over 2} (f_\theta(x) - \text{symlog}(y))^2 \\[5pt]
\hat{y} := \text{symexp}(f_\theta(x)) \\[5pt]
\text{symlog}(x) := \text{sign}(x) \ln(|x|+1) \\[5pt]
\text{symexp}(x) := \text{sign}(x) (\exp(|x|) - 1) $$

$$
\text{Derivation:} \\[5pt]
y := \text{sign}(x) \ln(|x|+1) \\[5pt]
\text{sign}(y) \cdot |y| = \text{sign}(x) \ln(|x|+1) \\[5pt]
\ln(|x|+1) \ge 0 \Rightarrow \text{sign}(y) = \text{sign}(x) \\[5pt]
\exp(|y|) - 1 =  |x| \\[5pt]
\text{sign}(y) (\exp(|y|) - 1) = \text{sign}(x) \cdot |x| = x
$$

- The `symlog` function compresses the magnitudes of both large positive and negative values. 
    - Unlike the logarithm, it is symmetric around the origin while preserving the input sign. 
    - This allows the optimization process to quickly move the network predictions to large values when needed. 
    - The `symlog` function approximates the identity around the origin so that it does not affect learning of targets that are already small enough.

---

- **For potentially stochastic targets, such as rewards or returns, we introduce the `symexp twohot loss`**
    - **Here, the network outputs the `logits` for a softmax distribution over `exponentially spaced bins` $b_i \in B$**
    - **Predictions are read out as the weighted average of the bin positions weighted by their predicted probabilities.** 
    - Importantly, the network can output any continuous value in the interval because the weighted average can fall between the buckets:

$$ \hat{y} := \text{softmax}(f(x))^T B \qquad B := \text{symexp}([ -20, ..., +20 ]) $$

- The network is trained on `twohot encoded` targets, a generalization of `onehot encoding` to continuous values. 
    - **The `twohot encoding` of a scalar is a vector with $|B|$ entries that are all $0$ except at the indices $k$ and $k + 1$ of the two bins closest to the encoded scalar.** 
    - The two entries sum up to $1$, with linearly higher weight given to the bin that is closer to the encoded continuous number.
    - The network is then trained to minimize the categorical cross entropy loss for classification with soft targets. 
    - Note that the loss only depends on the probabilities assigned to the bins but not on the continuous values associated with the bin locations, decoupling the size of the gradients from the size of the targets:

$$ L(\theta) := - \text{twohot}(y)^T \log \text{softmax}(f_\theta(x)) \qquad \text{Cross-Entropy} $$

- Applying these principles, Dreamer 
    - **transforms vector observations using the `symlog` functions**, both for 
        - **the encoder inputs** 
        - and **the decoder targets** 
    - and employs the `symexp twohot loss` for 
        - **the reward predictor**
        - and **critic** 

- **We find that these techniques enable robust and fast learning across many diverse domains.** 

- For critic learning, an alternative asymmetric transformation has previously been proposed, which we found less effective on average across domains. 
    - **Unlike alternatives, `symlog` transformations avoid** 
        - truncating large targets, 
        - introducing non-stationary from normalization, 
        - or adjusting network weights when new extreme values are detected.

## Results

- We evaluate the generality of Dreamer across 8 domains — with over 150 tasks—under fixed hyperparameters. 
    - We designed the experiments to compare Dreamer to the best methods in the literature, which are often specifically designed and tuned for the benchmark at hand. 
    - We further compare to a high-quality implementation of `PPO`, a standard reinforcement learning algorithm that is known for its robustness. 
    - We run PPO with fixed hyperparameters chosen to maximize performance across domains and that reproduce strong published results of PPO on [ProcGen](https://github.com/openai/procgen). 
    - To push the boundaries of reinforcement learning, we apply Dreamer to the challenging video game Minecraft, comparing it to strong previous algorithms. 
    - **Finally, we analyze the importance of individual components of Dreamer and its robustness to different model sizes and computational budgets.** 
    - All Dreamer agents are trained on a **single Nvidia A100 GPU** each, making it reproducible for many research labs. 
 
### Benchmarks 

- We perform an extensive empirical study across 8 domains that include 
    - continuous and discrete actions, 
    - visual and low-dimensional inputs, 
    - dense and sparse rewards, 
    - different reward scales, 
    - 2D and 3D worlds, 
    - and procedural generation. 
- Figure 1 summarizes the benchmark results, showing that Dreamer outperforms a wide range of previous expert algorithms across diverse domains. 
- **Crucially, Dreamer substantially outperforms PPO across all domains.**
    - **Atari**: This established benchmark contains 57 `Atari 2600` games with a budget of `200M` frames, posing a diverse range of challenges. We use the sticky action simulator setting. **Dreamer outperforms the powerful MuZero algorithm while using only a fraction of the computational resources. Dreamer also outperforms the widely-used expert algorithms Rainbow and IQN**
    - **ProcGen**: This benchmark of 16 games features randomized levels and visual distractions to test the robustness and generalization of agents. Within the budget of `50M` frames, Dreamer matches the tuned expert algorithm `PPG` and outperforms `Rainbow`. Our `PPO` agent with fixed hyperparameters matches the published score of the highly tuned official `PPO` implementation
    - **DMLab**: This suite of 30 tasks features 3D environments that test spatial and temporal reasoning. In `100M` frames, Dreamer exceeds the performance of the scalable `IMPALA` and `R2D2+` agents35 at `1B` environment steps, amounting to a data-efficiency gain of over 1000%. We note that these baselines were not designed for data-efficiency but serve as a valuable comparison point for the performance previously achievable at scale.
    - **Atari100k**: This data-efficiency benchmark contains 26 Atari games and a budget of only `400K` frames, amounting to 2 hours of game time. `EfficientZero` holds the state-of-the-art by combining online tree search, prioritized replay, and hyperparameter scheduling, but also resets levels early to increase data diversity, making a comparison difficult. Without this complexity, Dreamer outperforms the best remaining methods, including the `transformer-based IRIS` and `TWM` agents, the model-free `SPR`, and `SimPLe`
    - **Proprio Control**: This benchmark contains 18 control tasks with continuous actions, proprioceptive vector inputs, and a budget of `500K` environment steps. The tasks range from classical control over locomotion to robot manipulation tasks, featuring dense and sparse rewards. Dreamer sets a new state-of-the-art on this benchmark, outperforming `D4PG`, `DMPO`, and `MPO33`
    - **Visual Control**: This benchmark consists of 20 continuous control tasks where the agent receives only high-dimensional images as input and has a budget of `1M` environment steps. Dreamer establishes a new state-of-the-art on this benchmark, outperforming `DrQ-v2` and `CURL47`, which are specialized to visual environments and leverage data augmentation.
    - **BSuite**: This benchmark includes 23 environments with a total of 468 configurations that are specifically designed to test credit assignment, robustness to reward scale and stochasticity, memory, generalization, and exploration. Dreamer establishes a new state-of-the-art on this benchmark, outperforming Boot DQN and other methods. Dreamer improves over previous algorithms especially in the scale robustness category.

### Minecraft 

- Collecting diamonds in the popular game Minecraft has been a long-standing challenge in artificial intelligence. 
    - Every episode in this game is set in a unique randomly generated and infinite 3D world. 
    - Episodes last until the player dies or up to 36000 steps equaling 30 minutes, during which the player needs to discover a sequence of 12 items from sparse rewards by foraging for resources and crafting tools. 
    - It takes about 20 minutes for experienced human players to obtain diamonds. 
    - We follow the block breaking setting of prior work because the provided action space would make it challenging for stochastic policies to keep a key pressed for a prolonged time.
    - Because of the training time in this complex domain, extensive tuning would be difficult for Minecraft. 
    - Instead, we apply Dreamer out of the box with its default hyperparameters. 
    - As shown in Figures 1 and 5, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without using human data as was required by VPT20 or adaptive curricula. 
    - All the Dreamer agents we trained on Minecraft discover diamonds in `100M` environment steps. 
    - While several strong baselines progress to advanced items such as the iron pickaxe, none of them discovers a diamond.

### Ablations 

![](../imgs/03_DreamerV3_Ablations.png)

- **Figure 6: Ablations and robust scaling of Dreamer.** 
    - a: All individual robustness techniques contribute to the performance of Dreamer on average, although each individual technique may only affect some tasks. Training curves of individual tasks are included in the supplementary material.
    - b, **The performance of Dreamer predominantly rests on the unsupervised reconstruction loss of its world model, unlike most prior algorithms that rely predominantly on reward and value prediction gradients.** 
    - c, **The performance of Dreamer increases monotonically with larger model sizes, ranging from `12M` to `400M` parameters. Notably, larger models not only increase task performance but also require less environment interaction.** 
    - d, **Higher replay ratios predictably increase the performance of Dreamer.** Together with model size, this allows practitioners to improve task performance and data-efficiency by employing more computational resources.

- **In Figure 6, we ablate the robustness techniques and learning signals on a diverse set of 14 tasks to understand their importance.** 
    - The training curves of individual tasks are included in the supplementary material. 

- We observe that all **robustness techniques** contribute to performance, 
    - most notably the **KL objective of the world model**, 
    - followed by **return normalization** 
    - and **symexp twohot regression** for `reward` and `value` prediction. 

- In general, we find that each individual technique is critical on a subset of tasks but may not affect performance on other tasks.

- **To investigate the effect of the world model, we ablate the learning signals of Dreamer by** 
    - **stopping either the task-specific reward and value prediction gradients** 
    - **or the task-agnostic reconstruction gradients from shaping its representations.** 

- Unlike previous reinforcement learning algorithms that often rely only on task-specific learning signals, **Dreamer rests predominantly on the unsupervised objective of its world model. This finding could allow for future algorithm variants that leverage pretraining on unsupervised data.**

### Scaling properties 

- To investigate whether Dreamer can scale robustly, we train **6 model sizes ranging from `12M` to `400M` parameters**, as well as different **replay ratios** on Crafter and a DMLab task. 
- **The replay ratio affects the number of gradient updates performed by the agent.** 
- Figure 6 shows robust learning with fixed hyperparameters across the compared model sizes and replay ratios.

- **Moreover, increasing the model size directly translates to both higher task performance and a lower data requirement.** 

- **Increasing the number of gradient steps further reduces the interactions needed to learn successful behaviors.** 

- The results show that Dreamer learns robustly across model sizes and replay ratios and that its performance and provides a predictable way for increasing performance given computational resources.

## Previous work

- Developing general-purpose algorithms has long been a goal of reinforcement learning research.

- `PPO` is one of the most widely used algorithms and is relatively robust but 
    - **requires large amounts of experience** and 
    - often **yields lower performance than specialized alternatives.** 

- `SAC` is a popular choice for `continuous control` and leverages experience replay for data-efficiency, but in practice 
    - **requires tuning, especially for its entropy scale**, and 
    - **struggles under high-dimensional inputs**

- `MuZero` plans using a value prediction model and has been applied to `board games` and `Atari`, but the 
    - **authors did not release an implementation** and 
    - the algorithm **contains several complex components**, making it challenging to reproduce. 

- `Gato` fits one large model to **expert demonstrations** of multiple tasks, but is 
    - **only applicable when expert data is available.** 
    
- In comparison, Dreamer masters a diverse range of environments 
    - **with fixed hyperparameters**, 
    - **does not require expert data**, and 
    - **its implementation is open source.**

- Minecraft has been a focus of recent research. With `MALMO`, `Microsoft` released a free version of the successful game for research purposes. `MineRL` offers several competition environments, which we rely on as the basis for our experiments.  The MineRL competition supports agents in exploring and learning meaningful skills through a diverse human dataset. 
    - `Voyager` obtains items at a similar depth in the technology tree as Dreamer **using API calls to a language model** but operates on top of the `MineFlayer` bot scripting layer that was specifically engineered to the game and **exposes high-level actions.** 
    - `VPT` trained an agent to play Minecraft through **behavioral cloning based on expert data of keyboard and mouse actions** collected by contractors and **fine-tuning using reinforcement learning** to obtain diamonds using `720 GPUs for 9 days`. 
    - In comparison, Dreamer uses the MineRL competition action space to autonomously learn to collect diamonds from sparse rewards using `1 GPU for 9 days`, without human data.

## Conclusion

- We present the third generation of the Dreamer algorithm, a general reinforcement learning algorithm that masters a wide range of domains with fixed hyperparameters. 
    - Dreamer excels not only across over 150 tasks but also learns robustly across varying data and compute budgets, moving reinforcement learning toward a wide range of practical applications. 
    - Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch, achieving a significant milestone in the field of artificial intelligence. 
    - As a high-performing algorithm that is based on a learned world model, Dreamer paves the way for future research directions, including teaching agents world knowledge from internet videos and learning a single world model across domains to allow artificial agents to build up increasingly general knowledge and competency.

## Methods

### Baselines

- We employ the Proximal Policy Optimization (PPO) algorithm, which has become a standard choice in the field, to compare Dreamer under fixed hyperparameters across all benchmarks. 
    - **There are a large number of PPO implementations available publicly and they are known to substantially vary in task performance.** 
    - To ensure a comparison that is representative of the highest performance PPO can achieve under fixed hyperparameters across domains, we choose the high-quality PPO implementation available in the [**Acme framework**](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py) and select its hyperparameters in Table 1 following recommendations and additionally tune its epoch batch size to be large enough for complex environments, its learning rate, and its entropy scale. 
    - We match the **discount factor** to Dreamer because it works well across domains and is a common choice in the literature. 
    - We choose the [**IMPALA network architecture**](https://arxiv.org/pdf/1802.01561) that we have found performed better than alternatives and set the minibatch size to the largest possible for one **A100 GPU**. 
    - We verify the performance of our PPO implementation and hyperparameters on the ProcGen benchmark, where a highly tuned PPO implementation has been reported by the PPO authors. We find that our implementation matches or slightly outperforms this performance reference.

- For Minecraft, we additionally tune and run the `IMPALA` and `Rainbow` algorithms because not successful end-to-end learning from scratch has been reported in the literature. 
- **We use the Acme implementations of these algorithms**, use the same **IMPALA network** we used for PPO, and tuned the learning rate and entropy regularizers. 
- For all other benchmarks, we compare to tuned expert algorithms reported in the literature as referenced in the results section.

### Implementation

#### Experience replay 

- We implement Dreamer using a uniform replay buffer with an online queue. 
    - Specifically, each minibatch is formed first from non-overlapping online trajectories and then filled up with uniformly sampled trajectories from the replay buffer. 
    - We store latent states into the replay buffer during data collection to initialize the world model on replayed trajectories, and write the fresh latent states of the training rollout back into the buffer. 
    - While prioritized replay is used by some of the expert algorithms we compare to and we found it to also improve the performance of Dreamer, we opt for uniform replay in our experiments for ease of implementation.
    - **We parameterize the amount of training via the replay ratio. This is the fraction of time steps trained on per time step collected from the environment, without action repeat.** 
        - Dividing the replay ratio by the time steps in a minibatch and by action repeat yields the ratio of gradient steps to env steps. 
        - For example, a replay ratio of 32 on Atari with action repeat of 4 and batch shape 16 × 64 corresponds to **1 gradient step every 128 env steps**, or 1.5M gradient steps over 200M env steps.

#### Optimizer 

- We employ **Adaptive Gradient Clipping (AGC)**, which clips per-tensor gradients if they exceed 30% of the L2 norm of the weight matrix they correspond to, with its default $\epsilon = 10^{−3}$.
    - AGC decouples the clipping threshold from the loss scales, allowing to change loss functions or loss scales without adjusting the clipping threshold. 
    - We apply the clipped gradients using the **LaProp optimizer** with $\epsilon = 10^{−20}$ and its default parameters $\beta_1 = 0.9$ and $\beta_2 = 0.99$. 
    - LaProp normalizes gradients by `RMSProp` and then smoothes them by momentum, instead of computing both momentum and normalizer on raw gradients as **Adam** does. 
    - **This simple change allows for a smaller epsilon and avoids occasional instabilities that we observed under Adam.**

#### Distributions 

- The `encoder`, `dynamics predictor`, and `actor distributions` are mixtures of 99% the predicted softmax output and 1% of a uniform distribution to prevent zero probabilities and infinite log probabilities. 

- The `rewards` and `critic` neural networks output a softmax distribution over **exponentially spaced bins** $b \in B$ and are trained towards **twohot encoded targets**:

$$ k := \sum_{j=1}^{|B|} \delta(b_j < x) \\[5pt]
\text{twohot}(x)_i := \begin{cases}
|b_{k+1} - x| / |b_{k+1} - b_k| & i = k \\
|b_k - x| / |b_{k+1} - b_k| & i = k+1 \\
0 & \text{else} 
\end{cases} $$

- **The output weights of twohot distributions are initialized to zero to ensure that the agent does not hallucinate rewards and values at initialization.** 

- **For computing the expected prediction of the softmax distribution under bins that span many orders of magnitude, the summation order matters and positive and negative bins should be summed up separately, from small to large bins, and then added. Refer to the source code for an implementation.**

#### Networks 

- Images are encoded using stride 2 convolutions to resolution $6 \times 6$ or $4 \times 4$ and then flattened and are decoded using transposed stride 2 convolutions, with sigmoid activation on the output. 

- Important:
    - Vector inputs are **symlog transformed** and then 
    - **encoded and decoded using 3-layer MLPs.**
    - **The actor and critic neural networks are also 3-layer MLPs** and 
    - **the reward and continue predictors are 1-layer MLPs.** 
    - **The sequence model is a GRU with block-diagonal recurrent weights of 8 blocks** to allow for a large number of memory units without quadratic increase in parameters and FLOPs. 
    - **The input to the GRU at each time step is a linear embedding of the sampled latent $z_t$, of the action $a_t$, and of the recurrent state to allow mixing between blocks.**

### Benchmarks

#### Protocols 

- Summarized in Table 2, we follow the standard evaluation protocols for the benchmarks where established. 
    - Atari uses 57 tasks with sticky actions. 
    - The random and human reference scores used to normalize scores vary across the literature and we chose the most common reference values, replicated in Table 6. 
    - DMLab uses 30 tasks and we use the fixed action space. 
    - We evaluate at 100M steps because running for 10B as in some prior work was infeasible. 
    - Because existing published baselines perform poorly at 100M steps, we compare to their performance at 1B steps instead, giving them a 10× data advantage. 
    - ProcGen uses the hard difficulty setting and the unlimited level set. 
    - Prior work compares at different step budgets and we compare at 50M steps due to computational cost, as there is no action repeat. 
    - For Minecraft Diamond purely from sparse rewards, we establish the evaluation protocol to report the episode return measured at 100M env steps, corresponding to about 100 days of in-game time. 
    - Atari100k includes 26 tasks with a budget of 400K env steps, 100K after action repeat. 
    - Prior work has used various environment settings, summarized in Table 10, and we chose the environments as originally introduced. 
    - Visual Control spans 20 tasks with an action repeat of 2. 
    - Proprioceptive Control follows the same protocol but we exclude the two quadruped tasks because of baseline availability in prior work

#### Environment instances 

- In earlier experiments, we observed that the performance of both Dreamer and PPO is robust to the number of environment instances. 
- Based on the CPU resources available on our training machines, we use 16 environment instance by default. 
- For BSuite, the benchmark requires using a single environment instance. 
- We also use a single environment instance for Atari100K because the benchmark has a budget of 400K env steps whereas the maximum episode length in Atari is in principle 432K env steps. 
- For Minecraft, we use 64 environments using remote CPU workers to speed up experiments because the environment is slower to step.

#### Seeds and error bars 

- We run 5 seeds for each Dreamer and PPO per benchmark, with the exception of 1 seed for ProcGen due to computational constraints, 10 seeds for BSuite as required by the benchmark, and 10 seeds for Minecraft to reliably report the fraction of runs that achieve diamonds. All curves show the mean over seeds with one standard deviation shaded.

#### Computational choices

- All Dreamer and PPO agents in this paper were trained on a single Nvidia A100 GPU each. 
    - Dreamer uses the 200M model size by default. 
    - **On the two control suites, Dreamer achieves the same performance using the substantially faster 12M model, making it more accessible to researchers.** 
    - The replay ratio control the trade-off between computational cost and data efficiency as analyzed in Figure 6 and is chosen to fit the step budget of each benchmark.

### Model sizes

- To accommodate different computational budgets and analyze robustness to different model sizes, we define a range of models ranging from 12M to 400M parameters shown in Table 3. 
    - The sizes are parameterized by the model dimension, which approximately increases in multiples of 1.5, alternating between powers of two and power of two scaled by 1.5. 
    - **This yields tensor shapes that are multiples of 8 as required for hardware efficiency.** 
    - Sizes of different network components derive from the model dimension. 
    - The MLPs have the model dimension as the number of hidden units.
    - The sequence model has 8 times the number of recurrent units, split into 8 blocks of the same size as the MLPs. 
    - The convolutional encoder and decoder layers closest to the data use 16× fewer channels than the model dimension. 
    - Each latent also uses 16× fewer codes than the model dimension. 
    - The number of hidden layers and number of latents is fixed across model sizes. All hyperparameters, including the learning rate and batch size, are fixed across model sizes.

### Previous Dreamer generations

- We present the third generation of the Dreamer line of work. Where the distinction is useful, we refer to this algorithm as DreamerV3. 
    - **The DreamerV1 algorithm was limited to continuous control**,
    - **the DreamerV2 algorithm surpassed human performance on Atari**, and 
    - **the DreamerV3 algorithm enables out-of-the-box learning across diverse benchmarks.**

- We summarize the changes that DreamerV3 introduces as follows:
    - **Robustness techniques:** 
        - **Observation symlog,** 
        - **KL balance and free bits,** 
        - **1% unimix for all categoricals,** 
        - **percentile return normalization,** 
        - **symexp twohot loss** for the reward head and critic.
    - **Network architecture:** 
        - **Block GRU**, 
        - **RMSNorm normalization**, 
        - **SiLu activation**.
    - **Optimizer:** 
        - **Adaptive gradient clipping (AGC)**, 
        - **LaProp (RMSProp followed by momentum)**.
    - **Replay buffer**: 
        - **Larger capacity**, 
        - **online queue**, 
        - **storing and updating latent states.**

### Minecraft

## Supplementary material

# 3 2023 DreamerV3

- [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104)

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


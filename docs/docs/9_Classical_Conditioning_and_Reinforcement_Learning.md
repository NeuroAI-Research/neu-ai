# 9 Classical Conditioning and Reinforcement Learning

## 9.2 Classical Conditioning

- The Biology: From Pavlov to Dopamine
    - Classical conditioning is the process by which animals learn to associate stimuli with rewards or punishments
    - The Behavioral Foundation
        - **Unconditioned Stimulus (US):** A stimulus that naturally triggers a response, such as food
        - **Unconditioned Response (UR):** The natural reaction to a US, such as salivating when receiving food
        - **Conditioned Stimulus (CS):** A neutral cue (like a bell) that, through learning, begins to elicit a response
        - **Conditioned Response (CR):** The learned response to the CS, marking the animal's expectation of a reward

- The Neural Mechanism: Dopamine in the VTA
    - Computational models suggest that the **midbrain dopamine system**, specifically the **Ventral Tegmental Area (VTA)**, encodes the **prediction error ($\delta$)**
    - **Early Training:** Dopaminergic neurons fire vigorously upon receiving an unexpected reward
    - **Late Training:** After learning, neurons fire at the time of the **stimulus** (the predictor) but stay at baseline when the reward actually arrives, as it is no longer a "surprise"
    - **Omission:** If an expected reward is denied, dopamine activity is inhibited below its basal firing rate at the exact time the reward was predicted

- A. The Rescorla-Wagner Rule (The Delta Rule)
    - This model provides a concise account of how animals learn a linear prediction of reward based on the presence of stimuli
    - $v$: The expected reward
    - $u$: A binary stimulus variable (1 if present, 0 if absent)
    - $w$: The strength of the association (weight)
    - $\delta$: The prediction error
    - $\epsilon$: The learning rate

$$ v = w \cdot u \\[5pt]
\delta = r - v \\[5pt]
w \rightarrow w + \epsilon \delta u $$ 

- B. Temporal Difference (TD) Learning
    - TD learning **extends** the Rescorla-Wagner rule to account for the **timing** of rewards within a trial
    - The model treats $v(t)$ as a prediction of the **total future reward** expected from time $t$ until the end of the trial ($T$)

    $$ v(t) \approx \langle \sum_{\tau=0}^{T-t} r(t+\tau) \rangle $$

    - $v(t)$ is calculated using a discrete time version of a linear filter
    
    $$ v(t) = \sum_{\tau=0} w(\tau) u(t-\tau) $$

    - Because future rewards are unknown, the model uses the next step's prediction ($v(t+1)$) to calculate the error
    
    $$ \delta(t) = r(t) + v(t+1) - v(t) \\[5pt]
    w(\tau) \rightarrow w(\tau) + \epsilon \delta(t) u(t-\tau)$$

- Explaining Behavioral Phenomena

| Paradigm | Behavioral Result | Mathematical Explanation |
| :--- | :--- | :--- |
| **Blocking** | Pre-training stimulus $s_1$ prevents learning stimulus $s_2$ | Since $w_1 = r$, the prediction $v$ is already correct; thus $\delta = 0$ and $w_2$ never increases |
| **Extinction** | A learned response decays when the reward is removed | If $r=0$, then $\delta$ is negative, causing the weight $w$ to exponentially decay toward 0 |
| **Overshadowing** | One stimulus is learned more strongly than another when paired | This is modeled by giving the two stimuli different learning rates ($\epsilon$) |
| **Secondary Conditioning** | $s_2$ predicts $s_1$, which predicts reward. $s_2$ now evokes reward expectation | **TD Learning** explains this: the positive $\delta$ spike at the time of $s_1$ acts as a reward to drive the learning of $s_2$ |





## 9.3 Static Action Choice

- Biological Context: Bee Foraging
    - The primary biological model is a bee choosing between different colored flowers (blue and yellow) to find nectar
    - **Learning Preference:** Bees preferentially land on the flower color that delivers more reward
    - **Adaptability:** If reward characteristics are swapped, bees quickly adjust their preferences
    - **Risk Aversion:** Real bumblebees often prefer a "constant" flower (reliable $2~\mu l$ of nectar) over a "variable" flower that has the same mean reward but higher uncertainty
    - **Subjective Utility:** This risk-averse behavior is modeled mathematically by assuming bees value nectar based on a **concave utility function** rather than raw volume, meaning the "utility" of a variable reward is perceived as lower than a steady one

- Mathematical Framework: Stochastic Choice
    - The decision process is modeled as a **stochastic two-armed bandit problem**
    - The model bee does not choose a flower deterministically but follows a probability distribution
    - To decide between blue ($b$) and yellow ($y$), the model uses the **softmax distribution**:

    $$ P[b] = \frac{\exp(\beta m_b)}{\exp(\beta m_b) + \exp(\beta m_y)} $$

    - **Action Values ($m_b, m_y$):** These parameters represent the "value" or preference for each action, adjusted through learning
    - **Exploration-Exploitation Parameter ($\beta$):** This constant controls the randomness of choice:
        - **Large $\beta$ (Exploitation):** The bee almost always chooses the flower with the higher $m$ value
        - **Small $\beta$ (Exploration):** The bee's actions are more random, allowing it to sample different flowers to see if reward conditions have changed

- Two Learning Models (The "Actors")
    - The chapter compares two mathematical methods for how the bee updates its action values ($m$) based on rewards ($r$)

- A. The Indirect Actor
    - This model learns the **average reward** for each action
    - **The Delta Rule:** When a bee visits a flower and receives reward $r$, it updates the value ($m$) using a prediction error ($\delta$):
    
    $$ m \rightarrow m + \epsilon(r - m) $$

    - **Mechanism:** $m$ eventually tracks the average nectar volume $\langle r \rangle$
    - The policy is "indirect" because it is mediated by these estimated values

- B. The Direct Actor
    - This model bypasses estimating average rewards and instead adjusts $m$ values to directly **maximize the total expected reward**
    - **Stochastic Gradient Ascent:** The $m$ values are updated so that, over time, the change is proportional to the gradient of the expected reward
    - **Learning Rule:** When action $a$ is taken, the values are updated as:

    $$ m_{a'} \rightarrow m_{a'} + \epsilon(\delta_{aa'} - P[a'])(r_a - \bar{r}) $$
    
    - $\delta_{aa'}$ is $1$ if $a=a'$ and $0$ otherwise
    - The Error Term $(\delta_{aa'} - P[a'])$: This compares what you did ($\delta$) with what you expected to do ($P$). If you took an action you usually don't take (low $P$) and got a huge reward, the update to $m$ will be very large.
    - $\bar{r}$ is a reference reward like the mean reward
    - The Reward Term $(r_a - \bar{r})$: This determines the direction of the change. If the reward was better than the baseline ($r > \bar{r}$), you increase the probability of that action. If it was worse than the baseline, you decrease it.
    - Update All Weights: Unlike the Indirect Actor, the Direct Actor updates all action values on every trial, even for the flowers it didn't visit.
    - **Performance:** While theoretically sound, the direct actor often learns more slowly than the indirect actor and can struggle to adapt when reward characteristics swap




## 9.4 Sequential Action Choice

- The Mathematical Mechanics
    - The "actor-critic" model operates by interleaving two distinct learning steps: evaluating how good a state is (the critic) and improving the choice of actions (the actor)

- The Critic: Policy Evaluation
    - The critic estimates the total future reward, $v(u)$, for a location $u$. It updates its weight $w(u)$ using the **temporal difference (TD) learning rule**:

    $$ \delta = r_a(u) + \gamma v(u') - v(u) \\[5pt] 
    w(u) \rightarrow w(u) + \epsilon \delta $$
    
    - $\delta$ measures the difference between the actual outcome (immediate reward $r_a$ plus the value of the next state $v(u')$) and the expected outcome ($v(u)$)
    - **$\epsilon$:** learning rate
    - To account for time, rewards received later are weighted less using a discount factor $\gamma \in [0, 1]$

- The Actor: Policy Improvement
    - The actor decides which action to take (e.g., turning left or right) using a **softmax distribution**
    - The probability of choosing action $a'$ at location $u$ is determined by its action value $m_{a'}(u)$

    $$ m_{a'}(u) \rightarrow m_{a'}(u) + \epsilon(\delta_{aa'} - P[a'; u])\delta $$

    - **$\delta_{aa'}$:** This is 1 if action $a'$ was the one actually chosen, and 0 otherwise
    - **$P[a'; u]$:** The current probability of taking action $a'$
    - **$\delta$:** The same prediction error from the critic. If $\delta$ is positive (the result was better than expected), the action value $m$ for the chosen action increases, making that action more likely in the future

- Biological Implementation
    - **Dopamine Gating:** The $\delta$ term is believed to be represented by **dopamine neurons** in the VTA or substantia nigra. These neurons release dopamine onto the **striatum** (part of the basal ganglia)
    - **The Three-Term Rule:** Synaptic learning in the striatum is "gated" by this dopamine signal. The three terms are:
        1. The input (state vector $u$ from hippocampal place cells)
        2. The output (the action chosen by the actor cell)
        3. The reinforcement signal ($\delta$ provided by dopamine)
    - **Hippocampal Place Cells:** These provide the spatial "state vector" $u$ used in the equations to help the animal identify its current location in the maze

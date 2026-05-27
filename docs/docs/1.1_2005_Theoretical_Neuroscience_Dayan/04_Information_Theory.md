# 4 Information Theory

## 4.1 Entropy and Mutual Information

- The core problem: **Neural responses are "noisy."** If you show a neuron the same image twice, you might get 10 spikes the first time and 12 the second. We need a way to measure how much of that "clicking" is a meaningful signal versus just random biological background noise.

- Defining "Surprise" ($h$)
    - To measure information, we first need to measure **unpredictability** (surprise). 
    - **$r$**: A specific neural response (like a spike count).
    - **$P[r]$**: The probability of that response occurring.
    - If a response is guaranteed ($P[r]=1$), it tells you nothing new. If it's rare, it's highly informative. 
    - We use a **logarithm** so that if you have two independent neurons, their information simply adds together:

$$ h(P[r]) = -\log_2 P[r] $$

- Entropy ($H$): The Raw Capacity
    - **Entropy** is the average of that surprise across all possible responses. 
    - It represents the maximum amount of information a neuron *could* theoretically send.
    - **Maximum Entropy:** Occurs when all responses are equally likely (total unpredictability).
    - **Zero Entropy:** Occurs if the neuron does the exact same thing every single time.

$$ H = \sum_{r} P[r] \; h(P[r]) = \int dr \; p(r) \; h(p(r)) $$

- The Noise Problem ($H_{noise}$)
    - Just because a neuron has high entropy (lots of variability) doesn't mean it’s telling us about the world; it might just be noisy. 
    - To find the **Noise Entropy**, we look at the response $r$ **given** a specific stimulus $s$ (written as $P[r|s]$)
    - We then average this uncertainty over all stimuli:

$$ H_{noise} = \sum_{s,r} P[s] \; P[r|s] \; h(P[r|s]) $$

- Mutual Information ($I_m$):
    - The **Mutual Information** is what’s left of the total entropy after you subtract the noise. It is the "useful" information.
    - Recall identity $P[r,s] = P[s] \; P[r|s]$

$$\begin{aligned}
I_m &:= H - H_{noise} \\
&= \left( -\sum_{r} P[r] \log_2 P[r] \right) - \left( -\sum_{s} P[s] \sum_{r} P[r|s] \log_2 P[r|s] \right) \\
&= -\sum_{r} \left( \sum_{s} P[r,s] \right) \log_2 P[r] + \sum_{s,r} P[r,s] \log_2 P[r|s] \\
&= \sum_{s,r} P[r,s] \left( \log_2 P[r|s] - \log_2 P[r] \right) \\
&= \sum_{s,r} P[r,s] \log_2 \left( \frac{P[r|s]}{P[r]} \right) \\
&= \sum_{s,r} P[r,s] \log_2 \left( \frac{P[r,s]}{P[s]P[r]} \right) \\
&=: D_{KL}(P[r,s] \; || \; P[r]P[s])
\end{aligned}$$

- **If the stimulus and response are independent:** $P[r,s] = P[r]P[s]$, the fraction becomes 1, $\log(1) = 0$, and the information is zero.
- **If the response is perfect:** The mutual information equals the full entropy of the stimulus.

- The KL Divergence, $D_{KL}(P || Q)$, measures the "distance" or divergence between a "true" distribution $P$ and a "candidate" distribution $Q$:

$$ D_{KL}(P || Q) := \sum_{i} P[i] \log_2 \left( \frac{P[i]}{Q[i]} \right) $$

- The Biology of **Continuous** Firing Rates
    - In the real world, firing rates are continuous. This creates a mathematical "explosion" because a continuous number has infinite precision (and thus infinite entropy). 
    - To fix this, we introduce a resolution limit ($\Delta r$).
    - When you subtract the two entropies to find $I_m$, this "infinity" cancels out, giving us a clean integral for continuous signals:

$$ I_m = \int ds \int dr \, p[s]p[r|s] \log_2 \left( \frac{p[r|s]}{p[r]} \right) $$





## 4.2 Information and Entropy Maximization

- Maximizing Entropy for a Single Neuron
    - **Under a Maximum Firing Rate Constraint:** If a neuron has a limit $r_{max}$, the mathematical result (found using **Lagrange multipliers**) is that $p[r]$ should be **constant** (a flat distribution). This is known as **histogram equalization**

$$ p[r] = \frac{1}{r_{max}} $$
    
- **The Tuning Curve Solution:** To achieve this flat distribution of responses, the neuron’s tuning curve $r = f(s)$ must follow:

$$\begin{aligned}
\text{Conservation of probability:} & \quad p(r) dr = p(s) ds \\
\text{And:} & \quad p(r) = \frac{1}{r_{max}} \quad   r \in [0, r_{max}] \\
\text{So:} & \quad \frac{dr}{ds} = r_{max} \, p(s) \\
\text{Since } r = f(s) \text{, integrate:} & \quad f(s) = \int_{s_{min}}^{s} r_{max} \, p(s') \, ds' 
\end{aligned}$$

- Population Coding: Redundancy and Independence
    - When many neurons are involved, simply maximizing individual entropy isn't enough because they might all send the same information (redundancy)
    - Total population entropy is maximized when neurons are **statistically independent**
    - This leads to two conditions for an optimal "factorial code":
        - **Factorization:** The total probability is the product of individual probabilities: $p[r] = \prod p[r_a]$
        - **Probability Equalization:** Each neuron's response is individually optimized

- Frequency space: band pass filter
    - **Whitening**: boost high frequency (since low frequency dominate in natural scenes) to make the spectrum **flat like white noise**.
    - **Reducing Noise:** In the real world, high-frequency signals are weak and dominated by noise. A pure entropy maximizer would mistakenly "boost" this noise
    - $\kappa$: spatial frequency

$$ |D_s(\kappa)| \propto \underbrace{\kappa}_{ \text{Whitening Filter} } \cdot \underbrace{ e^{-\alpha \kappa} }_{ \text{Noise Filter} } $$

- Frequency to Space:  Inverse Fourier Transform 
    - The high-pass part (whitening) creates a sharp positive peak (the "Center").
    - The low-pass part (noise suppression) creates a broader inhibitory area around it (the "Surround")
    
$$ f(x) = A \cdot e^{-\frac{x^2}{2\sigma_{center}^2}} - B \cdot e^{-\frac{x^2}{2\sigma_{surround}^2}} $$

- The Physical Meaning: 
    - The "Center" $(\sigma_{center})$ is the neuron looking at a spot, and the "Surround" $(\sigma_{surround})$ is the neuron subtracting the average of the neighbors. By subtracting the neighbors, the neuron is **calculating the surprise** — it only fires if the center is different from the surroundings. 
    - This is the mathematical definition of **Predictive Coding**: don't report what you can already predict from the pixels next door.





## 4.3 Entropy and Information for Spike Trains

- Entropy Rates and Interspike Intervals (ISI)
    - Entropy in spike trains usually grows linearly with time, so it is reported as an **entropy rate** ($\dot{H}$), measured in **bits per second** or **bits per spike**
    - If we assume that the intervals between spikes (interspike intervals, $\tau$) are statistically independent, the entropy rate can be bounded by the probability density $p[\tau]$:

$$ \dot{H} \le -\langle r \rangle \int_{0}^{\infty} d\tau \, p[\tau] \log_2(p[\tau]\Delta\tau) $$

- For a homogeneous **Poisson process** (where spikes are truly independent), this becomes an equality:

$$ \dot{H} = \frac{\langle r \rangle}{\ln(2)}(1 - \ln(\langle r \rangle \Delta\tau)) $$

- The Direct Method (Strong et al. Scheme)
    - To account for correlations between spikes (where one interval affects the next), the "Direct Method" is used 
    - **Binning**: A spike train is divided into small bins of size $\Delta t$
    - **Binary Words**: Each bin is assigned a `1` (spike) or `0` (no spike). A sequence of duration $T_s$ becomes a binary "word" $B$
    - **Probability Calculation**: $P[B]$ is the probability of finding a specific binary sequence anywhere in the recording
    
$$ \dot{H} = -\frac{1}{T_s} \sum_{B} P[B] \log_2 P[B] $$

- The Extrapolation Challenge
    - For any finite dataset, as the sequence length $T_s$ increases, the number of possible binary patterns becomes too large to observe accurately, leading to a "sampling problem" 
    - **Linear Extrapolation**: To find the true entropy, researchers plot the calculated entropy against $1/T_s$ and extrapolate to the y-intercept (where $1/T_s = 0$, representing an infinite sequence)

- Measuring Mutual Information
    - **Noise Entropy ($\dot{H}_{noise}$)**: This is calculated by playing the *exact same stimulus* multiple times. Any variation in the neuron's response across these identical trials is considered noise
    - **Information Rate**: $\dot{H}_{total} - \dot{H}_{noise}$

# 3 Neural Decoding

## 3.1 Encoding and Decoding

- **The Core Problem: The Stochastic Bridge**
    - The brain doesn’t see the world directly; it only sees spikes. Because neurons are **stochastic** (the same stimulus produces different spike counts every time), a single response $r$ is ambiguous. Decoding is the statistical process of estimating the stimulus $s$ that most likely caused the observed response $r$.

- **The Biological Context**:
    - This chapter sets the stage for how sensory systems (like vision or touch) must deal with noise. It defines the "Ideal Observer" — a **mathematical benchmark** used to see how much information is actually available in a neural signal, **regardless of how the brain eventually uses it**.

- **The Prior $p[s]$:** The probability of a stimulus occurring in the environment.

- **The Likelihood $p[r|s]$:** The probability of a specific neural response given a stimulus (the "encoding" model).

- **The Evidence $p[r]$:** The total probability of seeing response $r$ across all possible stimuli:

$$ p[r] = \sum_{s} p[r|s] \; p[s] $$

- **Bayes’ Theorem (The Decoder):** The fundamental tool to find the probability of a stimulus given a response:

$$ p[s|r] = \frac{p[r|s] \; p[s]}{p[r]} $$





## 3.2 Discrimination

- **The Core Problem: Choosing Between Two Evils**
    - Discrimination is a "forced-choice" version of decoding. Instead of estimating a continuous value, the observer must decide between two options (e.g., Was the motion Up or Down?). 
    - The problem is the **overlap**: if the response distributions for "Up" and "Down" overlap, there is no perfect threshold; you must accept a trade-off between missing a signal and seeing a ghost.


- **The Biology: The Newsome-Movshon Experiment**
    - **The Stimulus:** Monkeys watch "Random-Dot Kinematograms." **Coherence** is the percentage of dots moving together.
    - **The Recording:** Researchers record from **Area MT** (motion-sensitive neurons).
    - **The Discovery:** By comparing the **Neurometric curve** (how well a single neuron discriminates) to the **Psychometric curve** (how well the monkey discriminates), they found that **a single neuron is often as accurate as the entire animal**.
    - **The "Anti-Neuron":** To simulate the choice, researchers compare a neuron preferring the "plus" direction with a hypothetical "anti-neuron" preferring the "minus" direction.

- **The Math (Signal Detection Theory)**

- **Discriminability ($d'$):** Quantifies how "separated" the distributions are in units of standard deviation ($\sigma$):

$$ d' = \frac{\langle r \rangle_+ - \langle r \rangle_-}{\sigma} $$

- **The Decision Rule:** A threshold $z$ is chosen. If $r \geq z$, we choose (+).
    - **False Alarm Rate ($\alpha$):** The probability of choosing (+) when the stimulus was actually (-):
    - **Hit Rate ($\beta$):** The probability of choosing (+) when the stimulus was actually (+):

$$ \alpha = P[r \geq z | -] = \int_{z}^{\infty} p(r|-) \, dr \\[10pt]
 \beta = P[r \geq z | +] = \int_{z}^{\infty} p(r|+) \, dr $$

- **ROC Curve:** A plot of $\beta$ vs $\alpha$ as $z$ varies. The **Area Under the Curve (AUC)** is the probability that the observer will correctly identify the stimulus in a two-alternative forced-choice (2AFC) task.

$$ \text{Let } r_+ \sim p(r|+) \quad \text{and} \quad r_- \sim p(r|-) \\[10pt]
P[\text{correct}] := P(r_+ > r_-) = \text{(derivation)} = \int_{0}^{1} \beta(\alpha) \, d\alpha = \text{AUC} $$

- **Likelihood Ratio ($L(r)$):** The mathematically optimal way to decide (Neyman-Pearson Lemma). If this ratio exceeds a certain value, the choice is (+):

$$ L(r) = \frac{p[r|+]}{p[r|-]} $$





## 3.3 Population Decoding

- Main Idea:
    - The core concept is that while individual neurons are variable and have limited selectivity, a **population** of neurons with overlapping tuning curves can represent information with high accuracy
    - Decoding is the process of estimating the original stimulus value based on the observed spike-count firing rates of these neurons

- Biological examples:
    - **Cricket Cercal System:** Crickets use four specific interneurons to sense the direction of air currents (wind). These neurons have **cosine tuning curves**, meaning their firing rate is proportional to the cosine of the angle between the wind direction and the neuron's preferred direction
    - **Monkey Primary Motor Cortex (M1):** Similar to the cricket, neurons in the monkey's M1 cortex encode the direction of arm movements using **cosine-like tuning**. However, unlike the cricket's four orthogonal neurons, M1 contains thousands of neurons with preferred directions pointing in all possible directions, creating a highly redundant representation.

- A. The Vector Method (Population Vector)
    - **For neurons with cosine tuning**, the stimulus (represented as a vector $\vec{v}$) can be estimated by a weighted sum of the neurons' preferred direction vectors $\vec{c}_a$
    - $r_0$: background rate

$$ \vec{v}_{pop} = \sum_{a=1}^{N} \left( \frac{r - r_0}{r_{max}} \right)_a \vec{c}_a $$ 

- B. Optimal Decoding: **maximize Bayes’ formula:**

$$ p[s|r] = \frac{p[r|s] \; p[s]}{p[r]} $$ 

- **Maximum Likelihood (ML) Estimation:** 
    - If the prior $p[s]$ is constant, we find the $s$ that maximizes $p[r|s]$
    - For a population with **Poisson variability** and **Gaussian tuning curves of equal width**, the ML estimate is simply the firing-rate weighted average of preferred values:

$$ s_{ML} = \frac{\sum r_a s_a}{\sum r_a} $$

- **Maximum A Posteriori (MAP) Estimation:** 
    - This method chooses the $s$ that maximizes $p[s|r]$, allowing the inclusion of prior knowledge about the stimulus distribution

- The accuracy of any decoding scheme is limited by the **Fisher Information** ($I_F(s)$) through the **Cramér-Rao bound**
    - **Fisher Information for Poisson Neurons:** (This shows that neurons contribute most to decoding accuracy where their tuning curves have the **steepest slope**, rather than where their firing rate is highest)

$$ I_F(s) = T \sum_{a=1}^{N} \frac{f'_a(s)^2}{f_a(s)} $$ 

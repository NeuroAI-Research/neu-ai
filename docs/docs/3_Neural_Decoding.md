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

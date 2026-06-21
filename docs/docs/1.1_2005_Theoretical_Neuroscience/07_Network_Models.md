# 7 Network Models

## 7.2 Firing-Rate Models

- Total Synaptic Current ($I_s$)
    - The total current delivered to the soma is modeled by convolving the presynaptic spike trains with a **synaptic kernel** $K_s(t)$, which describes the time course of a single synaptic response

- **Synaptic Current from Spikes:** 
    - For a presynaptic neuron $b$ with weight $w_b$ and firing times $t_i$
    - $\rho_b(\tau)$ is the neural response function (a sum of Dirac $\delta$ functions)

$$ w_b \sum_{t_i < t} K_s(t - t_i) = w_b \int_{-\infty}^t d\tau \, K_s(t - \tau) \rho_b(\tau) $$

- **Transition to Firing Rates:** 
    - By replacing $\rho_b(\tau)$ with the continuous firing rate $u_b(\tau)$, the total current from all $N_u$ inputs becomes:

$$ I_s = \sum_{b=1}^{N_u} w_b \int_{-\infty}^t d\tau \, K_s(t - \tau) u_b(\tau) $$

- **Differential Form:** 
    - If $K_s(t)$ is an exponential ($e^{-t/\tau_s}/\tau_s$), $I_s$ can be described by:
    - This equation shows that the current $I_s$ relaxes toward the weighted sum of inputs with a time constant $\tau_s$

$$ \tau_s \frac{dI_s}{dt} = -I_s + \sum w_b u_b = -I_s + \mathbf{w} \cdot \mathbf{u} $$

- Postsynaptic Firing Rate ($v$)
    - The firing rate is determined by an **activation function** $F(I_s)$ 
    - **Threshold Linear Function:** A common choice is the **ReLU**:
    - where $\gamma$ is the firing threshold

$$ F(I_s) = [I_s - \gamma]_+ $$

- **Steady-State Rate:** For constant inputs, the output rate is simply $v_\infty = F(\mathbf{w} \cdot \mathbf{u})$

- Firing-Rate Dynamics
    - There are two main ways to model how the rate $v$ changes over time:

- **Current-Driven Dynamics:** Assumes the firing rate follows the current instantaneously:

$$ \tau_s \frac{dI_s}{dt} = -I_s + \mathbf{w} \cdot \mathbf{u} \quad \text{with} \quad v = F(I_s) $$

- **Rate-Driven Dynamics:** Assumes the firing rate is a low-pass filtered version of the input, which is more common for network analysis:
    - Here, $\tau_r$ is a time constant often related to (but typically smaller than) the membrane time constant.

$$ \tau_r \frac{dv}{dt} = -v + F(\mathbf{w} \cdot \mathbf{u}) $$

- Network Architectures
    - These equations scale to larger populations using matrices:
    - **Recurrent Network:** Including a weight matrix $\mathbf{M}$ for connections between output neurons:

$$ \tau_r \frac{d\mathbf{v}}{dt} = -\mathbf{v} + F( \underbrace{\mathbf{W} \cdot \mathbf{u} }_{\text{Feedforward}} + \underbrace{\mathbf{M} \cdot \mathbf{v} }_{\text{Recurrent}} ) $$

- Recurrence
    - In the text (Biophysical/Rate Models): Recurrence refers to Lateral Connectivity or "Intra-layer" connections. It describes neurons in the same population (e.g., a cortical column) talking to each other. The focus is on how these connections amplify or suppress signals within the same moment in time to reach an equilibrium.
    - In modern ANNs (RNNs/LSTMs): Recurrence refers to Temporal Connectivity. It describes a layer talking to itself in the future. The focus is on Memory—carrying information from time-step $T$ to $T+1$.

- Continuously Labeled Networks
    - In previous equations, $v_i$ referred to the $i$-th neuron. 
    - In a Continuously Labeled Network, we assume neurons are arranged according to a property, such as:
        - Orientation: The angle of a line a neuron "likes" to see ($\theta \in [0, \pi]$).
        - Spatial Location: The $(x, y)$ coordinate on the retina or the body.
    - Instead of a vector of discrete rates, we now have a Function $v(\theta)$
    - In the discrete model, we used a sum $\sum M_{ij} v_j$
    - In the continuous model, this becomes:
    - $v(\theta')$: The activity of a neuron at position $\theta'$
    - $M(\theta, \theta')$: The strength of the connection between a neuron at position $\theta$ and a neuron at position $\theta'$
    - The Integral: This represents the total input to the neuron at $\theta$ from every other possible orientation in the entire population

$$ \tau_r \frac{dv(\theta)}{dt} = -v(\theta) + F\left( \int d\theta' [W(\theta, \theta')u(\theta') + M(\theta, \theta')v(\theta')] \right) $$

- Simulating 100 billion individual neurons is computationally nearly impossible. However, if we treat the cortex as a continuous sheet, we can use Calculus (specifically Integro-Differential Equations) to predict the behavior of the whole system without simulating every cell.
    - We can solve for Stationary States: Where the activity forms a "bump" (representing a memory or a specific perception).
    - We can predict Waves: How an itch or a visual flash "spreads" across the brain's surface.






## 7.3 Feedforward Networks

- Coordinate Transformations in Feedforward Networks
    - This chapter describes how the brain converts a **retinal coordinate** ($s$) into a **body-centered coordinate** ($s+g$) using a single layer of feedforward connections. This transformation is essential for reaching toward an object regardless of where the eyes are looking.

- Biological Foundation: Area 7a and Gain Modulation
    - In the posterior parietal cortex (Area 7a), neurons do not simply encode the location of a stimulus. 
    - Their response to a visual stimulus is "multiplied" or **modulated** by the gaze angle.
    - **$s$:** Position of the stimulus relative to the fixation point (retinal).
    - **$g$:** Angle of the eyes relative to the body midline (gaze).
    - **$u$ (Input Neuron):** A "gain-modulated" neuron whose response $f_u$ is defined by:
    - Where $\xi$ is the preferred retinal location and $\gamma$ is the preferred gaze angle.

$$ f_u(s, g) = \exp\left(-\frac{(s - \xi)^2}{2\sigma_{\xi}^2}\right) \times \text{sig}(g - \gamma) $$

- The Mathematical Solution: Population Integration
    - The network computes the coordinate transformation by summing the activity of a vast population of these input neurons. The steady-state response of an output neuron ($v_\infty$) is given by:

$$ v_{\infty} = F\left( \int d\xi d\gamma \, w(\xi, \gamma) f_u(s - \xi, g - \gamma) \right) $$

- **The "Shift" Constraint:**
    - To ensure the output represents a specific body-centered location (like $0^\circ$), the synaptic weights $w(\xi, \gamma)$ must be a function of the **sum** of the preferred angles:
    - This specific weighting ensures that if the eyes move by $+10^\circ$, the retinal peak must move by $-10^\circ$ for the neuron to continue firing, effectively "locking" the response to a fixed position relative to the body.

$$ w(\xi, \gamma) = w(\xi + \gamma) = \exp\left(-\frac{(\xi + \gamma)^2}{2\sigma_w^2}\right) $$

### ANN vs BNN

Here is a concise breakdown of the computational flows for $s + g$ (Target = Retina + Gaze).

- The ANN Flow: Algebraic Calculation
    - In an Artificial Neural Network, the flow is **symbolic**. It treats numbers as magnitudes.
    - **Input:** Two specific wires carry the numbers $10.0$ ($s$) and $20.0$ ($g$).
    - **Process:** A single neuron multiplies these inputs by weights (e.g., $1.0$) and sums them: $(10 \times 1) + (20 \times 1)$.
    - **Output:** One neuron fires with an **intensity** of $30.0$.
    - **Key Concept:** The **magnitude** of the signal represents the value.

- The BNN Flow: Topographic Mapping
    - In a Biological Neural Network, the flow is **spatial**. It treats numbers as physical addresses.
    - **Input:** No "number" is sent. Instead, a specific **location** on the Retinal Map (the $10^\circ$ spot) and a **location** on the Gaze Map (the $20^\circ$ spot) become active.
    - **Process (Gain Modulation):** The Gaze signal acts as a "routing switch." Based on the synaptic weights $w(\xi + \gamma)$, the gaze signal shifts the incoming retinal activity to a new destination.
    - **Output:** A specific **location** on the Body-Centered Map (the $30^\circ$ spot) becomes active.
    - **Key Concept:** The **address** of the active neuron represents the value.





## 7.4 Recurrent Networks

- Linear Recurrent Models
    - A basic linear recurrent network is described by a differential equation where the change in firing rate $v$ depends on the input $h$ and the internal feedback via a weight matrix $M$:
    - To solve this, the text uses **eigenvector expansion**. 
    - The firing rate vector $v(t)$ is expressed as a sum of eigenvectors $e_{\mu}$ of the matrix $M$:
    - This allows the system to be broken down into independent equations for each coefficient $c_{\nu}$:

$$ \tau_{r}\frac{dv}{dt}=-v+h+M\cdot v \\[5pt]
v(t)=\sum_{\mu=1}^{N_{c}}c_{\mu}(t)e_{\mu} \\[5pt]
\tau_{r}\frac{dc_{v}}{dt}=-(1-\lambda_{v})c_{v}(t)+e_{v}\cdot h $$

- Key Mathematical Behaviors:
    - **Stability:** If any eigenvalue $\lambda_{v} > 1$, the network is unstable (rates grow infinitely).
    - **Selective Amplification:** If an eigenvalue $\lambda_{1}$ is close to 1, the network massively amplifies the input component aligned with that eigenvector:    
    $$ v_{\infty}\approx\frac{(e_{1}\cdot h)e_{1}}{1-\lambda_{1}} $$
    - **Integration:** If $\lambda_{1} = 1$, the network acts as an integrator, maintaining activity even after the input $h$ is removed (used to model eye position):
    $$ v(t)\approx\frac{e_{1}}{\tau_{r}}\int_{0}^{t}dt^{\prime}e_{1}\cdot h(t^{\prime}) $$

- Continuous Linear Networks
    - For neurons labeled by a continuous variable (like a stimulus angle $\theta$), the summation becomes an integral:

    $$ \tau_{r}\frac{dv(\theta)}{dt}=-v(\theta)+h(\theta)+\rho_{\theta}\int_{-\pi}^{\pi}d\theta^{\prime}M(\theta-\theta^{\prime})v(\theta^{\prime}) $$

    - This is solved using **Fourier Series**, where the eigenvalues $\lambda_{\mu}$ correspond to different Fourier components of the weight function $M$:
    
    $$\lambda_{\mu}=\rho_{\theta}\int_{-\pi}^{\pi}d\theta^{\prime}M(\theta^{\prime})cos(\mu\theta^{\prime})$$

- Nonlinear and Rectified Networks
    - Since biological firing rates cannot be negative, a **rectification** function $[ \cdot ]_+$ is introduced:
    
    $$\tau_{r}\frac{dv(\theta)}{dt}=-v(\theta)+[h(\theta)+\frac{\hat{\lambda}_{1}}{\pi}\int_{-\pi}^{\pi}d\theta^{\prime}cos(\theta-\theta^{\prime})v(\theta^{\prime})]_{+}$$

    - This nonlinearity enables several advanced behaviors:
        - **Winner-Takes-All:** The network can pick the strongest of two competing inputs and suppress the other.
        - **Gain Modulation:** Adding a constant "gaze" input can multiplicatively scale (gain-modulate) the response to a visual stimulus.
        - **Sustained Activity:** Strong recurrent connections allow the network to "lock" into a pattern of activity that persists without input, serving as a form of **working memory**.

- Associative Memory
    - The text describes how to set synaptic weights $M$ so that specific activity patterns $v^m$ become fixed points of the network (Autoassociative memory). 
    - A common method is the **covariance rule**, which links units that are active together in a memory:
    
    $$ M=\frac{\lambda}{c^{2}aN_{b}(1-a)}\sum_{n=1}^{Nmcm}(v^{n}-acn)(v^{n}-acn)-\frac{nn}{aN_{b}} $$

    - This allows the network to "recall" a full memory pattern even if provided with only a noisy or partial starting input.

- Computational functionalities:
    - Selective Amplification
    - Integration
    - Winner-Takes-All
    - Gain Modulation
    - Sustained Activity
    - Associative Memory

- brain modules:
    1. The Oculomotor System (Neural Integrator)
        - **Mechanism:** **Linear Integration ($\lambda = 1$).**
        - This module is responsible for maintaining horizontal eye position. When the eyes move to a new position, the muscles require a constant tonic signal to counteract the elastic pull of the orbit. 
        - The network acts as a mathematical integrator: it receives a transient velocity pulse (input $h$) and, because it is tuned to have an eigenvalue of 1, it maintains a persistent firing rate (output $v$) even after the input ceases. This allows the eyes to remain fixed at a specific angle.
    2. Primary Visual Cortex (V1)
        - **Mechanism:** **Selective Amplification & Nonlinear Gain Modulation.**
        - **Orientation Tuning:** Recurrent connections amplify weak sensory inputs that align with a neuron's "preferred" orientation. This sharpens the response of the network to specific visual edges.
        - **Contextual Modulation:** The nonlinearity ($[ \cdot ]_+$) allows the network to change its "gain" (sensitivity). For example, if a visual stimulus is presented, but the animal is also performing a specific task or shifting its gaze, the recurrent feedback can multiplicatively scale the visual response without changing the preferred orientation of the neurons.
    3. Prefrontal Cortex (PFC)
        - **Mechanism:** **Sustained Activity / Persistent States.**
        - This is the biological basis for **Working Memory**. Unlike the oculomotor integrator which maintains a continuous range of values, the PFC uses strong recurrent excitation to create "stable fixed points." Once an input pushes the network into a specific state (e.g., remembering a specific location or a rule), the recurrent feedback is strong enough to keep the neurons firing in that pattern for several seconds, effectively "holding" the information in mind until it is needed.
    4. Hippocampus (CA3 Region)
        - **Mechanism:** **Associative Memory (Attractor Dynamics).**
        - The CA3 region is characterized by high levels of recurrent collaterals (neurons exciting their neighbors). It functions as an **autoassociative memory** using a covariance-based weight matrix (Hebbian learning). 
        - **Pattern Completion:** If a partial or noisy version of a memory is presented, the recurrent dynamics drive the network toward the nearest "stored" fixed point, effectively "completing" the memory.
    5. Head-Direction System
        - **Mechanism:** **Continuous Attractor Networks (Bump Attractors).**
        - This system uses a continuous version of recurrent math where the neurons are arranged conceptually in a circle (representing 0° to 360°). Recurrent excitation between "neighbors" in the head-direction space and inhibition for distant neurons creates a "bump" of activity. This bump represents the animal's current heading and is moved around the circle by vestibular inputs, allowing the brain to maintain a sense of direction even in the dark.





## 7.5 Excitatory-Inhibitory Networks

- The Two-Population Model (Homogeneous)
    - The simplest model describes all excitatory neurons with a single firing rate $v_E$ and all inhibitory neurons with rate $v_I$. The dynamics are defined by two coupled differential equations:

$$ \tau_{E}\frac{dv_{E}}{dt} = -v_{E} + [M_{EE}v_{E} + M_{EI}v_{I} - \gamma_{E}]_{+} \\[5pt]
\tau_{I}\frac{dv_{I}}{dt} = -v_{I} + [M_{II}v_{I} + M_{IE}v_{E} - \gamma_{I}]_{+} $$

- **Important Note on $M$:**
    - For this specific homogeneous model, **the synaptic weights ($M_{EE}, M_{IE}, M_{EI}, M_{II}$) are now scalars** rather than matrices
    - They represent the total strength of the connections between the populations
    - However, in more complex versions of E-I models (like the olfactory bulb or selective amplification models later in the chapter), $M$ can return to being a matrix or a function of spatial/preferred angles

- Phase-Plane Analysis
    - To understand the system's behavior, we look at the **phase plane** ($v_E$ vs $v_I$)
    - **Nullclines:** These are the lines where $\frac{dv_E}{dt} = 0$ or $\frac{dv_I}{dt} = 0$. For the equations above, these are straight lines
    - **Fixed Points:** The intersection of these nullclines represents a fixed point where the system remains static

- Linear Stability Analysis
    - The stability of a fixed point (whether the system returns to it after a disturbance) is determined by the **stability matrix**
    - **Stability Matrix ($L$):** Formed by taking derivatives of the rates at the fixed point:

$$ L = \begin{pmatrix} \frac{M_{EE} - 1}{\tau_E} & \frac{M_{EI}}{\tau_E} \\[5pt]
\frac{M_{IE}}{\tau_I} & \frac{M_{II} - 1}{\tau_I} \end{pmatrix} $$

- **Eigenvalues ($\lambda$):**
    - The stability is determined by the real parts of the eigenvalues of this matrix:

$$\lambda = \frac{1}{2} \left( \frac{M_{EE}-1}{\tau_E} + \frac{M_{II}-1}{\tau_I} \pm \sqrt{\left( \frac{M_{EE}-1}{\tau_E} - \frac{M_{II}-1}{\tau_I} \right)^2 + \frac{4M_{EI}M_{IE}}{\tau_E\tau_I}} \right)$$

- **Stable Fixed Point:** Real parts of both $\lambda < 0$
- **Unstable Fixed Point:** At least one real part of $\lambda > 0$
- **Oscillatory Behavior:** Occurs when the term under the radical is negative, making the eigenvalues a complex conjugate pair
    - **Collapsing Spiral:** Real part is negative
    - **Expanding Spiral/Limit Cycle:** Real part is positive

- Key Transitions: Bifurcations
    - **Hopf Bifurcation:** Occurs when a fixed point becomes unstable and a **limit cycle** (periodic oscillation) emerges as a parameter (like $\tau_I$) is changed
    - **Saddle-node Bifurcation:** Occurs when two fixed points (one stable, one unstable) collide and disappear

- Application Examples
    - **The Olfactory Bulb:** Uses an E-I model where $M_{EE} = M_{II} = 0$ (based on anatomy) to explain how odor inputs trigger 40 Hz oscillations during sniffs
    - **Selective Amplification:** Demonstrates that E-I networks can achieve high selective amplification without creating "spurious" tuned activity (persistent perceptions) that purely excitatory networks might suffer from





## 7.6 Stochastic Networks (Boltzmann Machine)

- The Input: $I_a(t)$
    - Before we get to probability, we need to know the "pressure" acting on a neuron. 
    - This is a simple weighted sum. It says the total input to neuron $a$ is the sum of external "nudges" ($h_a$) and the influence of all its neighbors ($M_{aa'} \cdot v_{a'}$)

$$ I_a(t) = h_a(t) + \sum_{a'=1}^{N_v} M_{aa'} \cdot v_{a'}(t) $$

- The Sigmoid "Switch": $F(I_a)$
    - In this model, neurons are binary: they are either **1 (On)** or **0 (Off)**. We use a probability rule to decide their state:
    - We use the **Sigmoid function** because it perfectly maps any input (from $-\infty$ to $+\infty$) onto a scale between 0 and 1 
    - If the input $I_a$ is very large and positive, $F(I_a)$ is nearly 1 (the neuron will almost certainly turn on). If it's very negative, $F(I_a)$ is nearly 0 (it stays off).

$$ P[v_a = 1] = F(I_a) = \frac{1}{1 + \exp(-I_a)} $$

- The Energy Function: $E(v)$
    - To understand the whole network, we define its **"stress level"** or **Energy**:
    - In physics, systems naturally want to fall into low-energy states. 
    - We put a **negative sign** in front so that if a neuron $v$ is "On" and its input $h$ or $M$ is positive (meaning it *should* be on), the Energy **decreases**. A low Energy score means the network is in a state that "makes sense" given its weights

$$ E(v) = -h \cdot v - \frac{1}{2}v \cdot M \cdot v $$

- The **Boltzmann Distribution**: $P[v]$
    - Because the network is stochastic (constantly "shaking" with noise), it doesn't just sit in one state. It wanders, but it prefers low-energy states
    - The exponential $exp(-E)$ turns **addition** (adding up energies of different neurons) into **multiplication** (the probability of independent events)
    - $Z$ is the **partition function**, which is just the sum of the top part for every possible state

$$ P[v] = \frac{\exp(-E(v))}{Z} $$



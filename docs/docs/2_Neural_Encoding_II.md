# 2 Neural Encoding II

## 2.2 Estimating Firing Rates

- **Linear-Nonlinear (LN) Model**:

$$ r_{est}(t) = r_0 + F\left( \int_{0}^{\infty} D(\tau) s(t - \tau) d\tau \right) $$

- $r_{est}(t)$: The estimated firing rate at time $t$
- $r_0$: The background firing rate when no stimulus is present
- $s(t - \tau)$: The stimulus history; how the stimulus behaved $\tau$ milliseconds ago
- $D(\tau)$: The **Linear Kernel** (or filter); it weights the stimulus history to find patterns the neuron "prefers"
- $\int \dots d\tau$: The **Linear Stage**; a convolution that measures the overlap (similarity) between the stimulus and the kernel
- $F(\cdot)$: The **Static Nonlinearity** (Activation Function); it transforms the linear match into a realistic firing rate by adding a threshold (ReLU) or saturation (Sigmoid)
- If you know the kernel $D$ and the nonlinearity $F$, you can **predict how the neuron will respond to any arbitrary stimulus** $s(t)$






## 2.3 Introduction to the Early Visual System

- The Retina
    - **Phototransduction:** Rod and cone receptors convert light into electrical signals via hyperpolarization
    - **Graded Potentials:** Horizontal, bipolar, and amacrine cells use smoothly changing membrane potentials to represent light intensity 
    - **Retinal Ganglion Cells (RGCs):** These are the output neurons of the retina. They convert graded signals into **action potentials** for long-distance transmission via the optic nerve
        - **ON cells:** Fire when light is turned on
        - **OFF cells:** Fire when light is turned off

- Visual Processing Stream: The signal travels from the **retina** $\rightarrow$ **optic chiasm** (where some axons cross to represent opposite visual fields) $\rightarrow$ **Lateral Geniculate Nucleus (LGN)** of the thalamus $\rightarrow$ **Primary Visual Cortex (V1)**






## 2.4 Reverse-Correlation Methods: Simple Cells

- Core Discovery: The Spatio-Temporal Filter
    - **V1 Simple Cell** acts as a linear-nonlinear filter. It doesn't just respond to a "pixel" at a single moment; it integrates light information across a patch of space and a window of time. 
    - **Spatial Selectivity:** The neuron "looks" for specific orientations (edges).
    - **Temporal Selectivity:** The neuron "looks" for changes in light (transients).
    - **Linear-Nonlinear (LN) Model:** The neuron first performs a linear weighted sum (convolution) of the input, then applies a static nonlinearity (rectification) to produce a firing rate.

- The Biology
    - **Anatomy:** V1 neurons receive input from the LGN. Their receptive fields are composed of parallel **Excitatory (ON)** and **Inhibitory (OFF)** zones.
    - **Symmetry:**
        - **Even-symmetric:** Prefers a light or dark **bar** in the center ($\phi = 0$ or $\pi$).
        - **Odd-symmetric:** Prefers an **edge** or transition ($\phi = \pi/2$).
    - **Temporal Adaptation:** Most V1 neurons are **biphasic**. If a stimulus doesn't move, the neuron stops firing. This prevents the brain from wasting energy on "old news."

- The textbook defines the linear response $L(t)$ as a 3D integral over space ($x, y$) and history ($\tau$):

$$ L(t) = \int_{0}^{\infty} d\tau \iint dx dy \, D(x, y, \tau) s(x, y, t-\tau) $$

- **The Spatial Kernel ($D_s$):** A Gabor function.

$$ D_s(x, y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp\left(-\frac{x^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2}\right) \cos(kx - \phi) $$

- **The Temporal Kernel ($D_t$):** A biphasic function.

$$ D_t(\tau) = \alpha \exp(-\alpha\tau) \left( \frac{(\alpha\tau)^5}{5!} - \frac{(\alpha\tau)^7}{7!} \right) $$

- The Fast-Slow Approximation (The "Live" View)
    - replace the complex $D_t$ polynomial with a **Difference of Leaky Integrators**. This mimics the "feedforward inhibition" found in biology.
    - **Fast Path (Excitation):** $ L_{f}(t) = r_f L_{f}(t-1) + (1-r_f) S_t $
    - **Slow Path (Inhibition):** $ L_{s}(t) = r_s L_{s}(t-1) + (1-r_s) S_t $
    - **Biphasic Result:** $ L(t) = L_{f}(t) - L_{s}(t) $
    - **Why this works:** The "Slow" path acts as a moving average of the background. By subtracting it, we effectively perform a **temporal derivative**, turning the neuron into a **Change Detector**.

- Final Processing: The Nonlinearity

$$ r(t) = [\text{ReLU}(L(t))]^2 \\[5pt]
\text{Spikes} \sim \text{Poisson}(r(t)) $$

| Concept | Biology | Math |
| :--- | :--- | :--- |
| **Edge Detection** | Gabor-like Dendrites | 2D Convolution |
| **Memory/Adaptation** | Synaptic Fatigue/Inhibition | Biphasic $D_t$ |
| **Firing Threshold** | Voltage-gated Channels | Rectification |
| **Noise** | Neurotransmitter release | Poisson Process |

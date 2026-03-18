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

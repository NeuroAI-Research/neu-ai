# 8 Plasticity and Learning

## 8.2 Synaptic Plasticity Rules

- The Foundation: Linear Firing-Rate Model
    - Before modeling how synapses *change*, we must model how the neuron *fires*. The text uses a simplified linear model for the postsynaptic activity ($v$):
    - Because synaptic changes are much slower than firing rate changes, we can assume the neuron reaches its steady state quickly

$$ \tau_{r}\frac{dv}{dt}=-v+w\cdot u \\[5pt]
\text{steady state: } v = w \cdot u $$

- The Basic Hebb Rule: "Cells that fire together, wire together"
    - The simplest rule for weight change is:
    - It mathematically captures Donald Hebb’s 1949 conjecture: if the input ($u$) and output ($v$) are active at the same time, the synapse ($w$) gets stronger
    - **The Problem (Instability):** This rule is mathematically "unstable". If you take the dot product of the rule with the weight vector ($w$), you find that the change in the length of the weight vector is $\tau_{w}d|w|^{2}/dt=2v^2$. Since $2v^2$ is always positive, the weights will grow infinitely

    $$ \tau_{w}\frac{dw}{dt} = vu $$

- Solving Unbounded Growth: **The Oja Rule**
    - The subtraction term is proportional to the current weight ($w$)
    - Mathematically, this forces the sum of the squares of the weights ($|w|^2$) to relax to a fixed value ($1/\alpha$)

$$ \tau_{w}\frac{dw}{dt} = vu-\alpha v^{2}w $$

- Timing-Based Rules (STDP)
    - The final refinement acknowledges that the *order* of firing matters
    - If the presynaptic spike ($u$) happens just *before* the postsynaptic spike ($v$), the synapse is likely responsible for the firing and is strengthened
    - If it happens *after*, it couldn't have caused the firing and is weakened
    - The function $H(\tau)$ defines this sensitive time window

$$ \tau_w \frac{dw}{dt} = \int_{0}^{\infty} d\tau [H(\tau)v(t)u(t-\tau) + H(-\tau)v(t-\tau)u(t)] $$ 

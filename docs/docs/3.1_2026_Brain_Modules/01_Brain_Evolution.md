# 1 Brain Evolution

- Inspired by the book *A Brief History of Intelligence*, instead of trying to understand the human brain straight away, it makes more sense to learn about modules in the brain following the path of evolution. We start with the simplest animal, and then investigate what new features were added to serve what purpose - one brain module at a time

- In addition, we think about how we can replicate these brain modules' **functionalities** using modern AI. (we do not try to make the replication 100% biologically realistic, we only focus on replicating the principles, following ideas in [Neuroscience-Inspired Artificial Intelligence](https://www.cell.com/action/showPdf?pii=S0896-6273%2817%2930509-3) by Demis Hassabis)


## 1 Flatworms

![](https://upload.wikimedia.org/wikipedia/commons/f/fc/Platyhelminthes_diversity.jpg){width=200}

![](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcQYhTG8gIKNSHMyXbaGDGbkWRHVvCzUdvgHCZJsPUIuyxAJtTOJLrhOyDZC2XYKi7dxBKjJJbzGDACdIzA){width=300}

- [Video](https://www.youtube.com/watch?v=m12xsf5g3Bo)

---

1. Sensory Input Tier (The Perimeter Sensors)
    - `Eyespots (Ocelli)`: Dual Photoreceptors. Detect light intensity and vector direction; route raw analog signals straight to the head.
    - `Auricles (Lateral Head Flaps)`: Chemoreceptors. The worm's "nose." These flaps are packed with chemical sensors that taste the water current to locate food or detect toxic gradients.
    - `Marginal Receptors`: Mechanoreceptors & Tactile Sensors. A dense ring of tiny touch-sensitive cells running along the entire rim of the worm's flat body, mapping physical obstacles and water vibrations.

2. Central Routing Tier (The Master Gatekeeper)
    - `The Cerebral Ganglion`
        - Integrates the overlapping data from the Eyespots, Auricles, and Touch receptors. It acts as an instant reflex-router, balancing competing priorities (e.g., "Hungry, but there is too much light; retreat").

3. Distributed Execution Tier (The Ladder Network)
    - `Longitudinal Nerve Cords`: 
        - Dual ventral data highways housing the `Central Pattern Generators (CPGs)` to automate swimming oscillations.
    - `Transverse Commissures`: Horizontal ring-rungs coordinating left-to-right mutual inhibition for seamless steering control.
    - `Peripheral Nerve Plexus`: A fine, net-like mesh of minor nerves branching off the main ladder out to the extreme edges of the tissue, distributing micro-control commands locally.

4. Motor Output Tier (The Actuators)
    - `Circular & Longitudinal Muscles`: 
        - Layered muscular sheets directly beneath the skin. 
        - `Circular muscles` squeeze to make the worm long and thin; 
        - `longitudinal muscles` contract to make it short and thick. 
        - Working together via the CPGs, they create the slithering undulations (蜿蜒起伏).
    - `Ventral Ciliated Epithelium`: A carpet of microscopic, hair-like cilia on the worm's belly. Controlled by low-level neural firing, they beat rhythmically against a trail of secreted mucus, allowing the worm to glide effortlessly without flexing a single major muscle.

---

- [Code (MuJoCo + RL, very rudimentary)](https://gymnasium.farama.org/environments/mujoco/swimmer)
    - [swimmer_v5.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/swimmer_v5.py)
    - [swimmer.xml (MuJoCo)](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/assets/swimmer.xml)

![](https://gymnasium.farama.org/_images/swimmer.gif){width=200}

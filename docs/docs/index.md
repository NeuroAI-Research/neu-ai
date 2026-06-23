# NeuroAI Research

- **Goal:** replicate (simulate) human brain **modules/circuits** using SOTA AI technologies

## Steps

1. `Learning` Learn the book [Theoretical Neuroscience](https://boulderschool.yale.edu/sites/default/files/files/DayanAbbott.pdf)
    - **Outcome:** Too general and vague, it talks about different mathematical models of neural networks, but not about how actual human brain **modules/circuits** work and achieve their functionalities

2. `Learning` Learn about SOTA AI technologies: [Transformers](https://arxiv.org/pdf/1706.03762)
    - Learn how SOTA AI achieve their remarkable abilities

3. `Learning` Curate a list of modules/circuits in the human brain identified in neuroscience

4. `Creating` Use SOTA AI technologies to replicate the functionalities of brain modules

## Code (Minimalist)

### Theoretical Neuroscience book (Dayan, 2005)

- [ann (25 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/ann.py)
- [1 neural encoding (103 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m1_neural_encoding.py)
- [2 neural encoding2 (209 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m2_neural_encoding2.py)
- [3 neural decoding (189 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m3_neural_decoding.py)
- [4 information theory (136 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m4_information_theory.py)
- [7 network models (252 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m7_network_models.py)
- [8 plasticity and learning (89 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m8_plasticity_and_learning.py)
- [9 reinforcement learning (158 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m9_reinforcement_learning.py)
- [10 representational learning (62 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/dayan2005/m10_representational_learning.py)


### LLM

- [2019 GPT2 (182 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/llm/y2019_GPT2.py)
- [2024 Gemma2 (254 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/llm/y2024_Gemma2.py)


### RL

- [1 RLBase (92 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/rl/m1_RLBase.py)
- [2017 PPO (119 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/rl/y2017_PPO.py)
- [2018 SAC (126 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/rl/y2018_SAC.py)
- [2023 DreamerV3 (180 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/rl/y2023_DreamerV3.py)


### SSL

- [2022 FlowMatching (62 lines)](https://github.com/NeuroAI-Research/neu-ai/blob/main/python/src/neu_ai/ssl/y2022_FlowMatching.py)

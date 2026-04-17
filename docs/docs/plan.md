Here's the **minimal study plan** — a focused, high-impact sequence of papers that gives you the **core science and essence** of 2026 SOTA MM-LLMs (vision-language, omni-modal, native fusion, MoE efficiency, advanced alignment, etc.).

This plan is deliberately small (8–10 papers total). It prioritizes **foundational ideas** first, then **evolution of architectures**, and finally **2025–2026 SOTA details**. Reading these in order, while taking notes on key mechanisms (attention variants, encoders, bridges, fusion strategies, training objectives), will let you understand **why** modern MM-LLMs work, not just what they do. Aim for 4–8 weeks if you read deeply and implement small examples.

### Phase 1: Foundations (Read these first — they explain the universal building blocks)
1. **Attention Is All You Need** (Vaswani et al., 2017) — arXiv:1706.03762  
   The original Transformer paper. Master self-attention, multi-head attention, positional encodings (including RoPE concepts), and decoder-only vs. encoder-decoder. This is the compute engine behind everything in 2026 models.

2. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2020) — arXiv:2010.11929  
   Introduces **Vision Transformer (ViT)** — how images become patch sequences. This is the basis for almost every vision encoder in Qwen3-VL, Gemma 4, Llama 4, etc.

3. **Learning Transferable Visual Models From Natural Language Supervision** (CLIP, Radford et al., 2021) — arXiv:2103.00020  
   Contrastive learning for aligning vision and language embeddings. This is the root of cross-modal alignment and projectors/bridges in early MM-LLMs.

### Phase 2: Classic MM-LLM Patterns (Understand the baseline before SOTA upgrades)
4. **BLIP-2** or **LLaVA** papers (e.g., "Visual Instruction Tuning", Liu et al., 2023 — arXiv:2304.08485)  
   The classic "ViT + projector/connector + frozen/LLM" recipe. See how modalities are bridged and why later models move beyond it. (Pick one; LLaVA is very readable.)

   Optional quick survey for context: **A Survey on Multimodal Large Language Models** (Yin et al., 2023/2024 updated — arXiv:2306.13549). Skim for taxonomy of architectures (dual-encoder, fusion, unified).

### Phase 3: 2025–2026 SOTA Technical Reports (The essence of current designs)
These directly describe the innovations in the models we discussed (Qwen3-VL, Gemma 4, Llama 4, etc.):

5. **Qwen3-VL Technical Report** (Bai et al., 2025) — arXiv:2511.21631  
   Deep dive into vision-language upgrades: interleaved-MRoPE for spatio-temporal modeling, DeepStack (multi-level ViT features), dynamic resolution, long interleaved contexts, and strong visual reasoning. Excellent on practical SOTA vision-first architecture.

6. **The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes** (Adcock et al., 2026) — arXiv:2601.11659 (or Meta's official blog on Llama 4)  
   Covers **native/early fusion** multimodality, MoE (interleaved dense/MoE layers, routed + shared experts), iRoPE for long/infinite context, and vision encoder based on MetaCLIP. This shows the shift to unified backbones.

7. **Gemma 4 Model Card + Technical Descriptions** (Google DeepMind, 2026)  
   Read the official model card (ai.google.dev/gemma/docs/core/model_card_4) and related blog posts. Focus on: hybrid attention (alternating sliding-window + global), variable-resolution vision tokens, Per-Layer Embeddings (PLE), MoE variants (e.g., 128 experts, few active), audio support on edge models, and efficiency tricks like shared KV cache and proportional RoPE.

### Optional but Highly Recommended for Completeness (Add if time allows)
- **Qwen3 Technical Report** (arXiv:2505.09388) — for the strong text backbone that powers Qwen3-VL.
- **STEP3-VL-10B Technical Report** (arXiv:2601.09668) — compact yet frontier-level example with language-optimized perception encoder and full unfreezing strategy.
- One recent survey update (e.g., 2024–2025 version of "A Survey on Multimodal Large Language Models" or BradyFU's Awesome-Multimodal-Large-Language-Models GitHub for latest taxonomy).

### How to Study Effectively (Minimal Effort, Maximum Understanding)
- **Read order**: Foundations → Classic → SOTA reports.
- For each paper: Focus on **architecture diagrams/sections**, attention/positional innovations, encoder + bridge details, training objectives (next-token on interleaved data), and ablation insights ("why this change matters").
- **Active learning**: After each foundational paper, sketch the mechanism (e.g., draw ViT patches → embeddings → attention). Compare how Qwen3-VL's DeepStack or Gemma 4's hybrid attention improves on the classics.
- **Implementation tip**: While reading, build/run tiny versions (e.g., a simple ViT + MLP projector + small Transformer) using PyTorch tutorials. This cements the "essence" faster than passive reading.
- Total reading time: ~20–40 hours if focused. Re-read sections on RoPE variants, fusion strategies, and MoE routing — these are the recurring "secret sauce" in 2026.

This set covers:
- Core Transformer science
- Vision encoding & alignment
- Evolution from bolted-on to **native early fusion**
- Efficiency (MoE, hybrid attention, token handling)
- SOTA specifics from leading 2026 models (Qwen, Llama, Gemma)

After these, you'll understand new papers quickly and see why techniques like MRoPE, variable-resolution, or sparse activation are used.

If you have limited time (e.g., only 4–5 papers), prioritize: **Attention Is All You Need** + **ViT** + **CLIP** + **Qwen3-VL** + **Llama 4 Herd** + **Gemma 4 model card**. That still gives ~80–90% of the essence.

Tell me your background (e.g., how comfortable you are with Transformers/math, time available, or if you want code resources alongside), and I can tweak this further or suggest exact sections to focus on!
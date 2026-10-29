# 4 2026 NeuroAI Follow-up

- [The recipe for intelligence in natural and artificial systems](https://www.cell.com/action/showPdf?pii=S0896-6273%2826%2900643-4)
    - DeepMind: `Christopher Summerfield`, `Kimberly Stachenfeld`

## Ricky's Critiques

> The entire `Large-scale generative models as theories of the brain` section, especially: 
> However, the AI breakthrough, when it came, was not inspired by either RL or symbolic cognition.

- Ricky's critique:
    - `Transformer / attention mechanism` are mainly for improving computation efficiency, instead of being the true source of intelligence. Switching `transformer` with `MLP` with same/larger amount of parameters can achieve the same abilities, only with much more compute.
    - The intelligence really comes from the training:
        - Supervised pre-training (next token prediction) turns the model into a cold `statistical token prediction machine`, it remembers facts, but has no real intelligence.
        - Intelligence really emerges from the RL phase when solving tasks.

> In their 2016 paper, the DeepMind team highlighted this capacity for "continual learning" as a key unsolved problem for AI. It remains unsolved today.
> In other words, like the renowned neuro-psychological patient HM, LLMs effectively have anterograde amnesia (顺行性遗忘症) during deployment — they are unable to encode new information into their long-term memory.

- Ricky's critique:
    - This is merely an engineering choice rather than "AI really lacks the ability of continual learning"
    - Technically, we can do this: during the day, we let LLM talk with users and save those conversations. At night, just like humans' sleeping, we pause the LLM app and use new data collected in the day to train (aka fine-tune) the model
    - Choosing not to do something doesn't mean it's "unsolved".

> They learn from dynamic interactions with the environment rather than passive exposure to content, making them creative in their dealings with an open-ended world.

- Ricky's critique:
    - The remarkable abilities from today's AI are exactly a result of RL training, which lets them interact with the environment 

---

## SUMMARY

- Ten years ago, two landmark papers discussed open problems in AI research from the standpoint of neuroscience and cognitive science. 
    - Today, many of these problems have been solved, mostly **without** direct inspiration from the study of the human brain. 
    - In this paper, we consider what is still shared between the twin endeavors of understanding natural brains and building artificial ones. 
    - We argue that, as well as understanding the "ingredients" of intelligence (its component parts), both fields are now coming to grasp the "recipe" for intelligence (how they fit together). 
    - Surprisingly, this recipe has many parallels in biological and artificial systems. 
    - Despite rapid progress in AI research, there are many computational problems that remain unsolved. 
    - We discuss how these problems may be addressed in natural intelligence.

## A LOOK BACK: NEUROSCIENCE AND AI IN 2016

- A decade ago, two landmark papers were published at the crossroads of neuroscience and AI research. 
    1. 2017 [Building machines that learn and think like people](https://arxiv.org/pdf/1604.00289) 
    2. 2017 [Neuroscience-Inspired Artificial Intelligence](https://www.cell.com/action/showPdf?pii=S0896-6273%2817%2930509-3)

- It was a heady time for both researchers interested in building AI and those seeking to understand biological brains. 
    - By 2016, deep neural networks had shown striking success in benchmark domains such as natural image recognition, machine translation, video, card and board games, and even TV gameshow quizzes. 
    - This hinted at a route to machine intelligence that combined large-scale neural networks with large-volume naturalistic training data. 
    - In tandem, evidence was starting to emerge that deep networks capable of striking behaviors might also serve as plausible models of biological brains, or at least of the primate neocortex.
    - Discussion of generally intelligent machines had begun to seep out of fringe techno-optimist communities and into the academic mainstream.
    - However, the idea that a machine could ever match the power of biological brains still seemed quite quixotic. 
    - The two landmark papers cautioned that there was a long way to go. 
    - "Building human-level general AI ... is a daunting task, because the search space of possible solutions is vast and likely only very sparsely populated," wrote a team from Google DeepMind in a paper published in this journal.
    - In a similarly cautious vein, authors from MIT wrote, "We should ... be mindful of what AI has achieved and what it has not. Although the pace of progress has been impressive, natural intelligence is still by far the best example of intelligence."

- Both papers advised that neuroscience and cognitive science offered a useful roadmap for building AI. 
    - Despite differences of emphasis, the two papers cited an overlapping list of unsolved challenges for AI research, drawing on themes that were familiar from the study of natural brains and minds. 
    - To emulate the power of human intelligence, they argued, future AI systems would need to form neural representations that supported strong forms of generalization and transfer — including the ability to compose new ideas from existing building blocks — instead of coming a cropper whenever faced with minor departures from their training distribution. 
    - They would need to evolve beyond "pattern recognition" systems and learn causal models of physical and psychological processes in the natural world — for example, how objects move and how people form beliefs. 
    - They would need to employ modular memory and attention systems over multiple timescales and support reasoning, planning, and imagination. 
    - The paper from the DeepMind team was, understandably, more bullish about deep learning, whereas the MIT team remained largely wedded to a nativist tradition, arguing that deep networks were overly data hungry and needed stronger inductive biases. 
    - Nevertheless, the two groups agreed that increased sample efficiency, and the consequential ability to learn and act on the fly, was a non-negotiable ingredient of intelligence that was yet to be realized in machines.

- In this review article, we discuss what the last decade of AI research means for psychology and neuroscience. 
    - **From the standpoint of 2026, the brain- and cognition-inspired roadmap proposed in these two papers might seem a little quaint (奇怪的,不合逻辑的,过时的).**
    - Today's multi-trillion-parameter foundation models have ridden roughshod (粗暴) over many of the unsolved challenges highlighted in these earlier papers. 
    - Frontier AI models can systematically compose new ideas and make wholly novel inferences, pick up new skills or knowledge on the fly, retrieve vast swathes (大片) of knowledge, flexibly solve problems, and intuitively grasp the intricacies (错综复杂) of the physical and psychological world. 
    - The backbone for these achievements is mastery of a cognitive feat that both papers were strangely shy of discussing — the ability to generate coherently structured outputs in natural language and other formal symbolic systems, such as mathematical notation and computer code.
    - This has turned today's AI models into remarkable tools for information retrieval, content generation, writing and coding assistance, problem solving, workflow orchestration, and robotic control. 
    - **It has also given impetus (动力) to a popular moral that `ever-growing model scale`, not `brain-inspired engineering`, is the key ingredient that underpins the success of AI systems.**
    - [**This was the message of a widely cited blog post arguing that the "bitter lesson" of AI research was that the field marched onward in tandem (协同) with Moore's law, as computers became faster and data more abundant, and not through the discovery of clever inductive biases.**](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)
    - **Did the papers by the DeepMind and MIT groups thus fail the test of time?**

- Here, we argue that they did not — at least not entirely. 
    - Both papers were instrumental in defining the unsolved puzzle pieces for building intelligent machines. 
    - Many of these puzzles now have solutions, in the form of new architectural and algorithmic innovations. 
    - **Although the answers were for the most part not directly inspired by biology, many are curiously synergetic (奇妙地协同) with classical ideas from psychology and neuroscience about how to build a brain.** 
    - In turn, this emerging picture of the building blocks of highly capable AI systems has shed new light on the computational processes that may power natural general intelligence.

## OUTLINE OF THIS PAPER

- In the sections below, we point to four major insights from AI research over the last decade and unpack their relevance for the understanding of natural intelligence:
    1. Large-scale generative models can acquire rich conceptual knowledge of the world from symbolic inputs (language and code) alone.
    2. Fast associative memory processes are essential for acquiring representations that underlie compositional and relational inferences — working out "what goes with what."
    3. Sample-efficient learning is possible by training systems to "learn how to learn," as long as neural systems explicitly represent the embedding space in which experiences are related.
    4. Successful planning and reasoning become much easier in systems with rich semantic knowledge acquired through extensive pre-computation.

- Nevertheless, many unsolved challenges remain. In a final section, we discuss major outstanding issues and propose insights from psychology and neuroscience that may be relevant. We focus on three shortcomings of modern AI systems:
    5. Current AI systems lack the ability to consolidate ongoing experience into long-term memory because their memory modules are poorly integrated.
    6. Meta-cognition and goal-directed behavior is unreliable in today's AI systems because they lack mechanisms for monitoring and self-modeling.
    7. Despite their superlative capacity for formal reasoning in verifiable domains, current models still struggle in open-ended and uncertain environments.

- We also acknowledge the possibility that the current paradigm may not be the final waystation (中转站) en route (在途中) to artificial general intelligence. 
    - Nevertheless, whereas both original papers emphasized the "ingredients" needed to build intelligence, we argue that both fields are starting to grasp the "recipe" for intelligence —  how component algorithms, architectures, and cost functions fit together to support intelligence in natural and artificial systems.

## PART 1: WHAT WE HAVE LEARNED IN THE LAST DECADE

- So much has changed since 2016; 
    - it is impossible to capture it all in a single article. 
    - In what follows, we try to give an overview of the most important advances and their relevance for cognitive science and neuroscience.

### Large-scale generative models as theories of the brain

- If we look back to 2016, the most promising paradigm in AI was deep reinforcement learning (RL), which had formed the bedrock of early successes in Atari and Go.
    - The RL paradigm assumes that some environmental states (such as the taste of an apple) emit special signals called rewards, and agents acquire intelligence by learning to maximize their expected reward. 
    - There are obvious reasons to consider RL a theory of the brain. 
    - It draws heavily on psychological theories of motivated behavior and has been successful at explaining neural activity, especially in subcortical circuits. 
    - Despite these synergies, using RL as a framework for understanding biological intelligence obliges us to overlook the problem that, in the natural world, its basic assumption does not hold. 
    - Rewards are not emitted by the environment (or even by the body) but are instead computed within the agent's brain (e.g., the taste of an apple is a sensory signal like any other, and it is your brain that decides it is delicious).
    - Nevertheless, in their paper, the DeepMind team leaned heavily on RL as a linking function between biological and artificial intelligence.

- The MIT team were influenced by the popular view that complex human behaviors are implemented by mental programs that involved discrete operations.
    - This was a familiar posture in cognitive science, which had long argued that intelligence involved a symbolic language of thought.
    - Proponents (支持者) of this position, however, had always struggled to explain how symbolic processing was acquired in the first place. 
    - The canonical (nativist) response — that it is inherited — displaces rather than solves this conundrum, as evolution itself relies on a different (but less efficient) form of learning. 
    - This had spawned a long and involved debate between symbolists and connectionists that has still not entirely faded.
    - In their paper, the MIT team took a compromise position, advocating for hybrid "neuro-symbolic" models, which blend discrete and connectionist approaches.

- **However, the AI breakthrough, when it came, was not inspired by either RL or symbolic cognition.** 
    - In 2017, hot on the heels of the two review articles, a new architecture for sequence modeling was proposed, called the **transformer**. 
    - Sequence modeling is a core aspect of the natural language processing (NLP) challenge of generating a plausible next word $x_{n+1}$ in a sentence $X = \{x_1; x_2; x_3 ... \; x_n\}$. 
    - **The transformer took a radical new approach to this problem, whereby chunks of data are ingested in a parallel, feedforward fashion, and the association strength between each input token (e.g., a word) and every other token is instantaneously computed in a giant computational graph (Figures 1A and 1B).** 
    - Being highly scalable, the transformer soon became the core building block of a new class of generative model called a large language model (LLM).
    - An early milestone was GPT-3, which captured the world's attention in 2020 for its ability to answer general knowledge questions and write long and (mostly) coherent passages of text.
    - Today, trillion-parameter transformer networks excel at tasks involving natural language, coding, math, and reasoning, as well as supporting multimodal content generation, digital tool use, and robotics.

- LLMs are generative models, which means that they learn by predicting new samples from their input distribution. 
    - The idea that this might describe how the brain works was a rather niche position in 2016. 
    - **The "active inference" framework proposed that the brain had evolved to infer the latent causes of sensation (recognition) and make predictions (generation) and that "reward" was really an epiphenomenon (副现象).**
    - There were suggestions that sensory systems might compute latent variables via the same principles as deep generative models.
    - The MIT group had advocated (提倡) for probabilistic inference as the fundamental computational engine for intelligence and provided proof-of-concept demonstrations of the power of probabilistic programs for learning causal models of the world and solving challenges such as recognizing objects and handwritten digits.
    - However, at the time, the idea that making and revising predictions about data might offer a universal solution for everything from generating valid sentences to winning at poker was not universally accepted.

- The arrival of LLMs taught us just how powerful generative models could be when scaled to billions of parameters. 
    - This has revived their cachet (威望) among neuroscientists. 
    - One example comes from a subfield of neuroscience that seeks to explain neural signals in the sensory neocortex as linear mappings from the activity patterns of task-optimized deep networks.
    - In its first flowering a decade ago, it looked to supervised learning in convolutional neural networks to explain visual cognition. 
    - However, recent approaches have relied on generative models, including variational autoencoders, contrastive models, and LLMs themselves as a tool of choice.
    - Transformer-based models have also been proposed as psycholinguistic theories and even used as foundation models for cognition. 
    - After decades of being derided as implausible and esoteric, predictive models are back in fashion.

- The success of LLMs has prompted neuroscientists to revisit other traditional pieties about computation and the brain. 
    - One thing we have learned is that, with sufficient data, weakly structured learning systems can learn highly structured behaviors. 
    - A neural network is such a system, with millions of interchangeable units and connections initialized at random, yet we now know that it can solve problems involving highly systematic operations, such as the correct use of nouns and verbs, long division, and logical syllogisms. 
    - Early neural networks showed hints of being able to learn symbolic tasks such as writing Python code, but the triumph of LLMs at math, coding, and reasoning has effectively laid to rest the long-running debate between symbolist and connectionist camps in cognitive science.

- Another surprising insight was just how much world knowledge is encoded in natural language. 
    - There were early hints that neural networks trained to predict the next word on large text-based corpora could represent high-level information about how the world is organized (Figures 1C and 1D). 
    - However, the dominant view, from philosophy to neuroscience, was that sensory experience and embodied action were non-negotiable requirements for an AI system to understand the natural world.
    - The DeepMind team bet on training deep networks to reproduce the verbal instructions that humans gave an agent navigating a 3D visual playroom environment. 
    - **However, early LLMs displayed some striking knowledge of a physical world they had never even seen. For example, transformers trained on text alone could infer the structure of color space and generalize an understanding of Euclidean geometry to unseen concepts. After reading a description of the layout of a house, GPT-4 was able to write a Python program that correctly visualized the map (Figure 2A).**
    - Critics were quick to point out that LLMs (and vision language models, or VLMs, that had also been jointly trained on images and their text descriptions) lagged behind humans on tests of real-world understanding, such as predicting whether a tower of blocks would fall over or navigating through a maze to a goal. 
    - Indeed, the idea that LLMs are fundamentally flawed because they lack a "world model" continues to carry currency.
    - **However, the view that models can learn meaning without reference to the physical world has become a popular, if not prevailing, view.**

- Despite astonishing recent progress in AI research, not everyone has embraced the view that transformer-based generative models are a promising paradigm for intelligence. 
    - Many still claim that AI systems cannot "really" be intelligent if they are merely predicting tokens.
    - Indeed, it is true that vanilla LLMs have several shortcomings that limit their readiness for universal, plug-and-play rollout. 
    - Model responses are not always reliable, content generation can be stereotyped and homogeneous, beliefs are often expressed overconfidently, reasoning can be faulty, and models often struggle to remain true to their goals over the long term.
    - In fact, as we discuss below, many of the inductive biases that were subsequently developed in AI research have served to paper over these cracks in basic LLM abilities. 
    - For example, naive transformers struggle with elaborate arithmetic calculations, but when equipped with temporally extended cognition, access to external tools, and a means to verify solutions during training (with RL), they can go on to almost win Math Olympiads and solve Erdos Problems.

### Fast associative processes for relational inference

- It is widely stated that the key to intelligence is relational inference — the ability to learn and exploit abstract patterns in data.
    - In psychology, analogical puzzles such as Raven's Progressive Matrices are the backbone of psychometric testing (Figure 2B). 
    - Neuroscientists have long searched for neurons that code for abstract concepts in a way that is invariant to their surface properties. 
    - In machine learning, the ability to transfer knowledge to out-of-distribution tasks with similar structure is a prized success metric.
    - This perspective was central to both review articles, which argued that future AI systems would need to learn generalizable abstractions and solve novel relational reasoning problems in the way that humans can.

- Despite this enjoinder, we do not know how humans and other animals learn about the relational structure of the world. 
    - The brain regions most commonly implicated are the prefrontal cortex (PFC) and the medial temporal lobe (MTL), including the hippocampus. 
    - Both contain neurons that code for rich abstractions.
    - **For example, neurons in the human MTL respond to famous individuals or buildings but are invariant to modality, and cells in the macaque hippocampus code for the spatial layout of an environment, irrespective (不论) of the landmarks present, or signal a relational "map" of values in a task. In the PFC, neurons code for the abstract rules that govern a task and fire to movement sequence archetypes independent of the actions required.**
    - However, the computations in mammalian MTL and PFC are quite different and support distinct types of memory.
        - The MTL is widely thought to associate the rich contextual features of a memory via Hebbian learning. This makes it crucial for laying down durable traces in an episodic store. 
        - By contrast, in the PFC, each neural state excites the next in a recurrent temporal chain. This allows it to integrate information and maintain latent states over a short delay period, making it vital for working memory.

- **A decade ago, state-of-the-art AI models were recurrent neural networks (RNNs) with learnable gates.**
    - **As hypothesized for the PFC, RNNs contain trainable parameters that map past onto current latent states, dictating how information is maintained, transformed, or discarded at each point in time. When combined with RL, systems based on RNNs led the field on benchmark challenges involving video games.**
    - RNN-based sequence-to-sequence models were used to model language and images for machine translation, language modeling, and image captioning. 
    - However, when asked to generate long passages of text, these models produced outputs that were superficially plausible, but the overall narrative was disappointingly incoherent.

- Transformers dispense with (免除) recurrence and model sequences in a wholly parallel, feedforward fashion (Figure 2C, left). 
    - Inputs $X$ are projected into a latent "value" space via a set of weights $W_v$.
    - However, additional trainable weights $W_k$ and $W_q$ map each token $x_i$ onto an indexable address (a "key" $k_i$) and a retrieval cue (a "query" $q_i$). 
    - For each token $x_i$, this allows values $v_j$ to be retrieved in proportion to how well the query $q_i$ matches the key $k_j$ for all other tokens $x_j$, according to a similarity kernel (e.g., a softmax function). 
    - Thus, this "self-attention" process allows the transformer to learn to encode relations — how each token in a sequence relates to every other and how these relationships determine future predictions. 
    - Representing tokens in a graph, rather than in a chain, turns out to be a crucial ingredient for modeling sentences. 
    - However, as transformers scale to hundreds of billions of parameters, and their context window (the length of $X$) grows to millions of tokens, they form a computational skyscraper that takes relational inference to new heights.
    - Superlative (最高级) scale allows LLMs to compute relations not just between words in a sentence or statements in a paragraph but also between the high-level narratives, arguments, or dependencies that exist across very long passages of text or code. 
    - This allows them to abstract over details and learn to predict outputs based on the most general and high-level structure in their input data.

- The magic mix of self-attention and scale has allowed LLMs to learn the sorts of abstractions that had eluded an earlier generation of deep networks (Figure 2C). 
    - **For example, using interpretability methods, GPT-2 was found to contain neurons that coded for linguistic concepts such as "verbs in the gerund form," "female names," or "words related to money."**
    - **When applied to image generation and captioning, a transformer-based model called Contrastive Language-Image Pretraining (CLIP) was found to contain units coding for multimodal semantic abstractions such as "Christmas" or "West Africa" — the AI equivalent of "concept neurons" in human MTL.**
    - **Researchers subsequently learned how to target concepts encoded in LLMs with techniques analogous to invasive brain stimulation. This allowed them to create a version of Claude (from Anthropic) that was obsessed with the Golden Gate Bridge and would bring it up in every conversation.**

- **Transformers are peculiarly effective at sequence modeling because parallel processing allows relations between each and every item to be computed instantaneously.** 
    - **However, in the natural world, time has the inconvenient property of being one-dimensional and irreversible, obliging animals to process sensory inputs one-by-one.** 
    - **In some ways, jettisoning (放弃) the RNN in favor of self-attention is a retrograde (逆行的) step for biological realism, and a significant departure from attempts to learn about intelligent computation from the biological brain.**

- Nevertheless, the transformer draws on computational principles with deep roots in psychology and neuroscience. 
    - Instead of recurrent memory, it uses fast associative learning to encode the past in a way that predicts the future. 
    - In psychology, this is called "episodic" memory, which (as in the transformer) has typically been modeled as an associative process that relies on the outer product of two sparse vectors encoding features of a memory. 
    - This computational scheme allows memories to be retrieved from noisy or partial inputs ("pattern completion"), especially when the retrieval process itself is iterative and governed by attractor dynamics.
    - In deep learning models, a rapid associative encoding step has long been proposed to help preserve memories against blurring by subsequent parameter tuning, and neuroscientists have proposed that the fast step occurs in the hippocampus, complementing slower learning in the neocortex.
    - A natural implementational scheme is Hebbian learning, variants of which allow neural feature selectivity to emerge after a single experience.
    - Softmax normalization in the transformer even confers sparsity, a capacity-enhancing transform thought to precede autoencoding in the MTL.
    - Thus, although the transformer cannot offer a complete model of biological computation, it relies on many of the principles of associative memory in the mammalian brain.

- As the brain is bound by time's arrow, it needs recurrent computation to make sure that the present moment is composed of the recent past. 
    - Indeed, numerous recording studies have shown that the geometry and dynamics of neural population activity during memory and inference tasks can be modeled with RNNs.
    - Recurrence allows multiple features of the past to linger into the present (e.g., maintaining a list in active memory).
    - This may serve as a "buffer," allowing fast associative processes in MTL to compute relations among items (e.g., knowledge of who likes whom). 
    - The brain employs other mechanisms to bring the past into the present for potential associative processing, such as experience-dependent replay, which may itself play an organizational role in memory.
    - Recent work has also shown that PFC neurons learn to code for current and proximal states in tandem, encoding an entire sweep of future states at once.
    - **Thus, although the transformer "cheats time" by processing its context in parallel, the computational principles on which it relies may be recreated in the brain in the form of highly scalable associative memory circuits.**

### Acquiring new knowledge on the fly

- A common critique of deep networks circa (大约) 2016 was their limited flexibility. 
    - Humans can grasp the rules of a new card game, make sense of a foreign sentence, or write a script in computer code.
    - This often involves picking up new skills on the fly, applying existing rules to unseen data, or combining familiar procedures to solve a novel problem. 
    - These capacities have traditionally been thought to require the encoding, composition, and manipulation of symbols. 
    - A decade ago, this view seemed vindicated by the failure of neural networks (that learn incrementally via slow weight updates) to demonstrate this sort of on-the-fly learning. 
    - It is still commonly lamented that neural networks are terribly sample inefficient and cannot match human versatility when it comes to learning new tasks. 
    - The success of the transformer, however, has offered us fresh ideas about how biological systems might deal flexibly with novelty.

- The secret is that, given scale and heterogeneous training, $W_q$ and $W_k$ can be adjusted so that self-attention "learns how to learn" within its inputs (called "in-context learning" or ICL; see Figure 3A). 
    - Like other neural networks, a transformer can memorize specifics via weight updates: for example (if planning a dinner party), that Alice likes to sit next to Charlie rather than Bob.
    - However, after diverse experience, transformer networks can learn more general principles — that young people prefer to sit together or that conservatives and liberals rarely mix. 
    - Faced with entirely new data, the transformer can apply these principles as if operating a mental program to deal with previously unseen guests of different ages and political persuasions. 
    - This can occur "on the fly" and without any weight updates, mirroring how humans pick up new ideas from just a few experiences. 
    - Transformers trade off in-weight and in-context learning in ways that are reminiscent of human memorization and generalization.

- ICL is not unique to transformers.
    - It is closely related to meta-learning in RNNs, whereby models learn generalizable patterns among diverse sequences of stimuli, actions, and outcomes and eventually converge to minima in weight space that allow on-the-fly solution discovery within their recurrent dynamics.
    - This has been proposed as a model of the PFC, and deep meta-RL systems behave like monkeys and humans on tasks involving multi-step inference, retrospective revaluation, meta-control, and learning set.
    - However, there is no doubt that self-attention is well suited to ICL, especially when training data is formatted such that parallel information is presented in a common template form, akin to an analogy.
    - When exposed to large quantities of relational data, transformers can learn generalizable syntactic rules in natural language, such as the agreement between direct objects and verbs or possessive pronouns and their matching noun phrases, and these are encoded in a segregable fashion in the attention heads of transformer models. 
    - This explains how transformers can master the infinite variety of natural language without ever being taught the rules of syntax, a feat that computational linguists had long thought would be impossible.

- Over the last decade, neuroscientists have also devoted increased attention to understanding how animals learn generalizable rules. 
    - The central challenge is the same in vivo as in silico.
    - If we simply memorize how Alice, Bob, and Charlie are related, we will not be able to generalize relational knowledge to new situations. 
    - Instead, we need to grasp that, for any set of items $x$, $y$, and $z$, the relations $\{x \to y\}$ and $\{y \to z\}$ constrain the possibilities for $\{x \to z\}$. 
    - Building on earlier thinking in cognitive science, a major idea that has emerged over the past decade is that neurons may explicitly encode positions in the embedding space in which stimuli and tasks reside.
    - Inputs are related by virtue of how they are "embedded" in a manifold whose axes encode dimensions of potential similarity, such as that given by the physical geometry of the world (here vs. there), time (now vs. later), observable features (big and red; small and blue), or a latent property (nephew or aunt). 
    - Neurons that signal how the similarity space itself is organized allow us to compute how examples $\{x \to y\}$ and $\{y \to z\}$ constrain responses to the "zero-shot" query $\{x \to z\}$. 
    - For example, the geometry of physical space in the natural world is Euclidean and low dimensional, so you know that if Xavier is next to Yolanda and Yolanda to Zac, then Xavier and Zac cannot be too far apart. 
    - Importantly, you can make this inference instantly without any prior learning about the people concerned — you have "learned how to learn" about how $x$ and $z$ are related from other contextual information.

- **An example cell type that codes for the embedding space itself (rather than the stimuli that populate that space) is found in the macaque dorsal stream.** 
    - **There, neurons have been observed to signal the number of items that should be remembered across a delay, irrespective of their physical form or modality (dots vs. flashes).**
    - More generally, the parietal cortex is a good candidate for encoding the structure of one-dimensional or "magnitude-like" embedding spaces in a way that underpins our knowledge of transitive relations (if x > y and y > z then x > z), accounts for our tendency to overgeneralize one-dimensional axes (e.g., to assume that "bigger" is "better"), and supports simple arithmetic calculation, which is impaired after parietal damage.

- Another celebrated example is the grid cell, found in the medial entorhinal cortex (mERC), which encodes periodically recurring positions on a 2D manifold (with the quirk that it is circularly wrapped into a toroidal form). 
    - This resembles how the coordinate grid lines on a map signal how Cartesian space is arranged, except in the mERC the symmetry is sixfold (hexagonal) rather than fourfold (quadrilateral). 
    - **Unlike its Nobel Prize co-winner, the hippocampal place cell, grid cells do not remap in novel environments and thus give the animal a consistent sense of how allocentric space is organized, irrespective of which landmarks it contains.** 
    - This is the sort of representation that would allow a rodent to infer that by running cheekily all the way around the edge of the dinner table, it would eventually get back to the place where it started (known as "path integration"). 
    - Similar periodic, hexagonal patterns of activity have even been (indirectly) observed in the medial PFC when people navigate more abstract 2D spaces, defined by continuously varying sensory features or latent values. 
    - Thus, the brain seems to explicitly encode not just the contents of experience but also the structure of the embedding space in which experience occurs, allowing it to make accurate predictions about new experiences that conform to this structure.

- Transformers are naively invariant to permutation (a fixed shuffle applied to the inputs would not perturb processing, as long as the attentional structure is maintained). 
    - Thus, to learn temporal relations in a sequence (e.g., that words that occur close together in a paragraph are usually more related), tokens are tagged with additional inputs called positional embeddings.
    - Initially, these were auxiliary sinusoidal signals added to token embeddings that varied smoothly across the context window, but in modern LLMs, position is usually encoded via gradual rotations of key and query matrices (Figures 3B and 3C).
    - At first, position codes might sound like a mere engineering trick. 
    - However, the gradual rotation means that self-attention will implicitly encode the relative distance between two tokens, so position codes share with grid cells both the logic of representing position in terms of phases and enabling vector computation and the periodic (oscillatory) nature of the positional code. 
    - More generally, the multiplexing of "what" (tokens) and "where" (position) helps transformers explicitly code for space or time in much the same way that grid or number cells explicitly encode the structure of the embedding space in which stimuli interrelate — a key part of the recipe for intelligence in both biological and artificial agents.

- A recent computational model of the mammalian MTL makes this explicit.
    - Called the Tolman-Eichenbaum machine (or TEM), it models navigation (through either physical or abstract spaces) by multiplexing neural signals for objects and estimated self-location using a fast associative process. 
    - It can thus be queried with locations (to predict the objects present where I am) or with objects (to elicit a self-location estimate via path integration from observed landmarks) and meta-learns effectively. 
    - In TEM, the neural codes for "position" are learned, rather than added by hand, as they are in the transformer. 
    - Critically, during training, they converge onto the same representational patterns found in the MTL — such as coding for space in a periodic (and sometimes sixfold) fashion and for the incident angle to objects or boundaries. 
    - In fact, with some subtle modifications, it is possible to make a version of the transformer with fully learnable position codes, thereby bridging the gap between transformers and TEM.
    - Other related accounts have similarly proposed that ERC might act as a positional "scaffold" for memories, and, together, **these models have revived the older idea that the hippocampus and neocortex store "keys" and "values" for associative memory, respectively, exactly like in the transformer.**

### Planning and reasoning with rich semantics

- RL requires agents to learn both the value of states of the world and how they are interconnected. 
    - The former allows "model-free" choices based on cached estimates of expected return, and adding the latter allows "model-based" search through a tree of possible futures to reach a goal.
    - In 2016, one of the hottest questions in neuroscience was exactly what blend of these strategies animals use.
    - Neural theories of model-based inference drew heavily on ideas from classical planning in AI, in which the main challenge was to find sensible heuristics for pruning a search tree, such as neglecting likely dead ends.
    - This mental search process was often argued to involve the lateral PFC, where lesions lead to deficits of planning and which is activated during multi-step choice and navigation.
    - In AI research, the combination of representation learning and tree search had been successful in building AlphaGo and in solving high-dimensional game environments where the mechanics were inferred but not given to the agent.

- The new lesson from the LLM era is that many classic reward-guided planning problems can be solved with semantic retrieval in natural language. 
    - Consider the canonical problem of choosing a restaurant. 
    - Theories grounded in model-based RL assumed that people mentally explored a tree of possible states until promising destinations were reached, backing up value estimates of quality and price in order to choose a left or right turn out of the door. 
    - However, LLMs—equipped with massive world knowledge — simply know that venue A had a horrible reputation and venue B was closed on Mondays and can make plausible recommendations by sampling from their generative model.
    - Confusingly, they can do this without ever explicitly computing reward. 
    - This is because people talk ceaselessly about their preferences, which then finds its way into the training data, allowing LLMs to emulate reward-based computation by simply generating likely token sequences. 
    - This has led some researchers to wonder whether many multi-step problems are actually solved in advance via pre-computation and meta-learning rather than relying exclusively on search or planning heuristics.

- When LLMs first came to prominence, a vigorous debate erupted over whether they were actually "thinking."
    - The models could often solve multi-step inference problems involving calculation or deduction, including complex reasoning puzzles. 
    - Suspiciously, however, their performance was often better when the problem was familiar, prompting claims that they were just parroting the answers encountered during training.
    - In fact, direct computation in the transformer activations does allow LLMs to solve unseen reasoning problems — but only if they are relatively simple. 
    - For example, raw arithmetic capabilities of vanilla LLMs break down as the numbers involved become too large. 
    - Although this finding was originally used to denigrate LLMs, it is actually what we might expect if they were behaving just like humans. 
    - When teaching our children math, for example, we provide them with props such as pen and paper, a calculator, and maybe even a teacher. 
    - Similarly, for LLMs, externalization and instruction are important ingredients for superlative reasoning. 
    - **In fact, unlike older LLMs, models employing chain-of-thought reasoning are provably Turing complete.**

- One example of externalization is to allow the LLM to use a mental scratchpad, where interim answers can be temporarily stored but not submitted to the user. 
    - Just as a human might articulate reasoning steps with an inner voice or jot down their working on paper, this allows the LLM to bootstrap toward the final solution by outputting tokens (called "chain-of-thought" reasoning — and models using this approach are often called LRMs, or large reasoning models).

- Another is tool use. 
    - Faced with complex calculations, today most people use a calculator or write a script in computer code. 
    - Modern LRMs have been trained to map quantitative and reasoning problems to API calls or Python code, which can then be executed in an interpreter, vastly improving their performance over naive calculation.
    
- We have also learned that for models to be highly effective reasoners they need to be explicitly trained how to think. 
    - Developers post-train models to use chain-of-thought to solve complex math and reasoning problems with verifiable solutions and use RL to update the weights.
    - This encourages the model to generate chain-of-thought outputs that are more likely to lead to a correct solution — similar to the teaching of mathematical reasoning or critical thinking in human education.

- In psychology and neuroscience, planning and reasoning have mostly been studied in simple and stylized settings. 
    - This has yielded theoretical frameworks that are semantically impoverished (贫困的) because, if you are planning in a two-step task or grid world, language is not strictly necessary to solve the task.
    - Recent work has begun to study planning in more complex settings, such as games, and to measure how humans "think aloud" during problem solving.
    - One remarkable finding is that neuroimaging data during game play can be explained an order of magnitude better by LRMs than by older AI systems (based on RL or Bayesian reasoning) that do not attempt to understand the task semantics (Figure 4).
    - This implies that neural alignment may be dramatically improved if we take into account the semantic structure of human knowledge.
    - Another recent study found that neural alignment during image recognition was significantly improved by incorporating semantic knowledge about objects derived from an LLM.
    - Results like these are changing how neuroscientists think about reasoning and planning.

## PART 2: UNSOLVED PROBLEMS FOR INTELLIGENT AGENTS

- Despite remarkable progress in building AI, much remains to be done. 
    - In this section, we discuss the challenges for AI development that are unsolved in 2026 and consider whether ideas from neuroscience and psychology can still help.

### Continual learning and memory consolidation

- In humans and other animals, experience-dependent plasticity occurs across the lifespan, so learning can continue into older age. 
    - This allows for forms of recursive self-improvement, whereby agents seek out experiences that will sculpt neural pathways in ways that will advantage them in the future (which is why we value curiosity, exploration, and a culture of education). 
    - **In their 2016 paper, the DeepMind team highlighted this capacity for "continual learning" as a key unsolved problem for AI. It remains unsolved today.** 
    - In almost all modern AI systems, parameter tuning is restricted to an initial training phase (in LLMs, typically comprising separate pre-training and post-training steps) and then turned off when the network is actually used. 
    - In other words, like the renowned neuro-psychological patient HM, LLMs effectively have anterograde amnesia (顺行性遗忘症) during deployment — they are unable to encode new information into their long-term memory.

- To compensate, two approaches are used. 
    - The first is to make the context window very long, giving LLMs a short-term memory system with superhuman capacity. 
        - However, this is a very inefficient strategy, as the computational cost of self-attention (inferring how each token relates to every other token) grows quadratically with context length. 
        - It also has important limitations, because retrieval from very long contexts can be unreliable.
        - For example, just like humans, LLMs show primacy (首要性) and recency (近期性) biases, and information can get "lost in the middle" of their context window.
    - The second approach is to write information to a nonparametric memory store.
        - This is a sort of cheat sheet containing information that it is useful to carry over from one context window to the next.
        - This trick, called "retrieval-augmented generation," is reminiscent (使人联想起的) of the idea from neuroscience that episodic information is retrieved during value computation, offering a (hippocampal-dependent) "third way" alongside model-free and model-based RL.
        - However, there is some way to go in training retrievers to solve classic memory dilemmas — what to store, for how long, and how to find it again.

- In biological systems, experiences provoke plastic changes that are initially fragile but become stronger as the most useful information is consolidated into long-term memory.
    - Consolidation relies on the neural reinstatement of experiences in a spatio-temporally coherent format during periods of rest and **sleep**, known as "replay."
    - Over the past decade, our understanding of this neural mechanism has matured. 
    - We now know that replay is not confined to the hippocampus and supports more than just spatial memory — it occurs across the brain and for content as diverse as words and motor skills.
    - Replay also seems to function as a prioritization mechanism, selecting information that is unfamiliar, novel, rewarded, or salient for encoding into long-term memory. 
        - For example, weak memories are more likely to be replayed in blood-oxygen-level-dependent (BOLD) signals, which, in turn, makes them stronger.
    - This selection may occur as the most relevant experiences are "tagged" for later consolidation.
    - Replay may also be a constructive process, linking diverse experiences to allow the composition of new ideas.
    - Biological systems, thus, are not confined by the hard-and-fast boundary that separates in-context and in-weight learning in LLMs. 
    - Instead, consolidation mechanisms allow them to flexibly move memories between short- and long-term repositories, ensuring that the most interesting and relevant memories are stored and made available for recall.

- Continual learning is hard for deep learning systems because the effect of new parameter tuning is typically either too large or too small relative to that of old learning. 
    - When new learning is too impactful, old memories get lost. 
    - This is called catastrophic interference, a longstanding problem to which neural networks are susceptible but biological systems are largely immune.
    - **Curiously, however, very large transformer-based networks are significantly more resistant to catastrophic interference than their earlier counterparts.** 
    - This may be because excess neural capacity allows new information to be encoded in a subspace that is largely orthogonal to existing knowledge, minimizing competition among gradient updates for old and new learning.
    - The converse problem is that the impact of continual weight updates on the network can be too weak. 
    - One reason this occurs is that neural networks trained with gradient descent gradually become less plastic, and so new inputs have diminished opportunity to affect downstream behaviors.
    - Regularization schemes that keep network weights small, or intermittently reinitialize small pools of neurons, seem to help maintain plasticity, but these have not yet been deployed at scale in production LLMs.

- Curiously, recent research into continual learning has highlighted surprising overlaps in the ways that humans and transformer networks learn. 
    - For example, some training interventions that benefit human learning, such as auto-correlating task practice in time (learning French on Monday and Spanish on Tuesday rather than studying both at once) are similarly advantageous in transformer networks, even though they disrupt learning in traditional neural networks.
    - Humans and transformer networks also learn to both memorize and generalize under the same data distributional properties.
    - Loss of plasticity is also a feature of healthy human aging, with certain functions acquired during early critical periods and older people showing reduced flexibility in the face of new learning.
    - We do not know whether these surprising parallels in how humans and transformers learn imply that continual learning may ultimately be solvable in LLMs, but there is currently no shortage of promising research ideas.

### Modeling and monitoring the self

- In 2016, the idea that AI might have its own intentions, purpose, or sense of self sounded like either a philosophical thought experiment or a Hollywood plotline. 
    - Today, it is a serious concern in AI safety.
    - AI systems can now autonomously pursue goals over extended periods — for as much as ∼20 h by recent estimates.
    - This has raised questions about the computational basis for one of the most elusive cognitive functions — the ability to maintain a consistent purpose over time or to construct a "sense of self."

- **In primates, the executive system is housed in the lateral portion of the PFC, and its primary role is to convert abstract plans into concrete actions.**
    - As discussed above, LRMs may be able to recreate this capacity for goal-directed behavior using chain-of-thought reasoning.
    - However, the neighboring medial aspect of the PFC (and underlying anterior cingulate cortex) plays a complementary role in monitoring ongoing task accuracy, estimating the value of exerting control, and computing the resource levels required for success.
    - This division of labor between an executive (responsible for selecting actions) and a monitor (responsible for exerting control) is a ubiquitous computational motif, even in non-biological information processing systems such as governments and corporations.
    - The monitoring function helps us remain committed to our goals and thus pursue strategic objectives over longer time horizons.
    - **Neural recording studies have suggested that the medial PFC makes the decision to persist with or abandon a goal, and lesions (损伤) to this area make humans and monkeys more distractible (易分心的).**
    - **A somewhat grander way of conceptualizing this division is that the lateral PFC encodes a model of the external environment, whereas the medial PFC learns a model of the agent itself.**
    - **Psychologists have long argued that a self-model is required to align our past experiences with our future goals and vice versa.** 
    - This helps us stay true to temporally extended objectives, such as building a career or raising a family, without being blown off course by distraction or adversity.

- Although AI researchers have worked hard to improve AI planning and reasoning about the external world, self-modeling has been less of a focus. 
    - This oversight causes widely discussed pathologies in LLM behavior.
        - For example, because LLMs do not adequately monitor their own performance, they often have a poor sense of what is possible and tend to give uncalibrated answers, meaning that their confidence and accuracy are only weakly related (which is one reason why models "hallucinate"). 
    - Although human meta-cognition is far from perfect, studying how we think about our own thinking may be an important puzzle piece for building more reliable AI.
    - AI systems often fail to persist with prior goal commitments, which means that during chain-of-thought reasoning they are prone to forget their original instructions and drift off into repetitive and unproductive loops.
    - Unlike healthy people, today's frontier AI systems do not, for the most part, have a stable persona or consistent value system but can be prompted to "role play" any character at will.
    - These deficiencies occur because AI systems lack a self-monitoring function designed to ensure that behavior remains on track.

- What would be needed to build AI systems that can behave purposively over the long term? 
    - Biological agents are motivated by drives, such as those that compel us to seek out sustenance and status.
    - This includes a drive to learn, allowing animals to exhibit curiosity, take active steps to learn about their environment, and recursively self-improve.
    - Many AI safety researchers have imagined that as AI systems grow more capable, they will naturally form drives similar to those of humans and will seek to perpetuate themselves or (more scarily) attempt to gain power for themselves.
    - This assumes that when an LLM describes its intentions in natural language (either in chain-of-thought or public responses), this is equivalent to it actually holding those intentions. 
    - Although we do not currently know whether this is the case or not, in biological agents, forming (or stating) an intention and acting on that intention can be sharply dissociated. 
        - For example, patients suffering from chronic apathy (慢性冷漠) (a disorder that often involves the medial PFC or the subcortical regions that project to it) often harbor very clear goals relating to work, relationships, or personal care but fail to act upon them because it doesn't seem worth the bother.
    - This implies that specific motivational mechanisms are needed to endow agents with the will to transform their stated intentions into purposive actions.

- Nevertheless, modern AI agents can sometimes behave as if they have their own intentions and purpose. 
    - They can autonomously discover and implement solutions to intricate problems, such as downloading a software library to help design a website or (in a fictional scenario) blackmailing a human to avoid being shut down.
    - Interestingly, their rapidly advancing capacity for long-form tasks (which may be doubling every few months, at least for cyber capabilities) may be due less to the increasing capabilities of foundation models and more to improvements in the suite of prompts, tools, and control functions wrapped around a model — typically referred to as the agent "harness."
    - **The harness typically implements a control loop, which monitors task execution and determines rules for resource allocation and stopping.** 
    - It thus implements functions that resemble those performed by the primate medial PFC. 
    - It may be that by optimizing the harness (including with gradient descent), developers are building an architectural innovation that will eventually permit more sophisticated forms of self-monitoring in AI systems.

### Discovery in open-ended environments

- Accounts of the blistering (极快的) pace of AI progress tend to focus on models' remarkable skills in formal domains. 
    - Frontier AI systems can find hidden software vulnerabilities, prove mathematical theorems, and write production-level code. 
    - However, the domains where progress has been most impressive are quite atypical of the natural world. 
    - Math and coding obey well-defined rules with exact, verifiable outcomes that throw errors when violated.
    - They can be solved by composing, chaining, or scripting many primitive operations into complex programs, which can then be executed without corruption from irreducible stochasticity in the environments. 
    - By contrast, many of the problems that biological agents face are less well defined. 
    - Most everyday challenges in private or professional life do not have clear success criteria, lack overt (明显的) feedback, involve noisy or ambiguous variables, or require you to interact with unreliable individuals or systems. 
    - When AI systems are tested on these types of problems, for example, in more open-ended benchmarks, they tend to fail or need substantial hand-holding from humans.
    - This is one reason why the promise of superhuman AI assistants has yet to be realized in practice.

- The limited robustness of AI systems is not a new problem.
    - Both review articles from 2016 noted that when recognizing images or playing video games, deep networks can be perturbed by seemingly inconsequential changes to the input distribution.
    - **In LLMs and VLMs, many of these earlier problems have been solved by scale alone.** 
    - Very large models, exposed to gargantuan (巨大的) datasets of text and images, have been trained on such diverse content that it is hard to push them out of distribution. 
    - Today, however, as the ambition of realizing fully automated AI assistants comes into closer focus, a more demanding sort of robustness is required. 
    - Current agentic AI systems can interface with external digital environments by writing code, browsing the web, sending messages, and controlling software.
    - This means that they are required to deal with the confusing, uncertain, and idiosyncratic aspects of the human digital world — ambiguous messages, broken hyperlinks, buggy applications, poorly designed interfaces, and unresponsive correspondents. 
    - AI systems are not yet able to behave intelligently in these types of open-ended environment that people navigate successfully every day.

- Negotiating open-ended environments requires finding creative solutions. 
    - Creativity is an ill-defined cognitive concept, but generating remote associations, divergent ideas, or atypical solutions is often cited as a core criterion. 
    - Unfortunately, modern LLMs tend to produce outputs that are overly generic, formulaic, or homogeneous. 
    - When asked to imitate people or describe scenarios, LLMs often show a phenomenon called "mode collapse," in which they prefer stereotypical answers rather than sampling from the full distribution of possible replies.
    - On tests of divergent thinking, such as generating semantically diverse associations or writing interesting stories, they tend to lag behind people, although there is growing evidence for gains from human-AI co-creativity.
    - As AI systems are increasingly used for scientific research (a classic open-ended domain) it is possible that their difficulty with generating genuinely novel ideas may limit their ability to make genuinely novel discoveries.

- Animals learn differently from current AI models. 
    - Instead of being trained via passive exposure to huge volumes of content, they are born with bodies that allow them to explore the environment.
    - Biological intelligence is thus shaped by the bidirectional relationship between actions and their consequences.
    - This view is called enactivism (生成认知论), and it has a long pedigree (血统) in cognitive science.
    - In AI research, RL proposes that intelligent agents learn to maximize reward by acting and observing in an agent-environment loop. 
    - It is not alone in emphasizing the reciprocity between actions and observations, with other theories proposing that intelligent agents should maximize empowerment (degrees of freedom over the environment) or minimize surprise (free energy) by taking actions that improve their model of the world (called "active inference").
    - Currently, a popular view is that current AI systems will never be truly innovative because they are only ever trained to generate samples from their input distribution, which consists only of existing human knowledge.
    - Thus, the prospect of true creativity, such as the notoriously surprising but decisive "move 37" in AlphaGo's championship match against Lee Sedol, may be impossible without a framework that allows AI systems to explore the world.

- Other theories of human creative discovery emphasize its social nature.
    - People have different experiences, leading to diversity in our collective knowledge, skills, and outlook. 
    - This allows communities to be more innovative than individuals, as the group generates highly varied ideas, and only the best ones are selected.
    - Most frontier AI systems today are hosted centrally and interact directly with users through millions of distinct consumer instances. 
    - There is thus no obvious way for discovery and innovation to emerge via cumulative machine culture. 
    - However, that is something that may yet occur as AI systems are deployed more widely and multi-agent interactions become routine.

## CONCLUSIONS

- The two 2016 papers discussed here predicted that neuroscience and psychology would be significant drivers of future progress in AI research. 
    - In practice, there were many other factors that proved more important — most notably, the emergence of **the transformer, an effective and highly scalable form of associative memory.** 
    - Nevertheless, the function of natural intelligence offers a helpful lens through which we can understand the successes and limitations of current AI systems. 
    - The last 10 years have brought us AI systems that can learn abstract patterns in data, acquire novel tasks on the fly, compose new outputs from existing building blocks, and reason powerfully about cause and effect. 
    - **These advances have been due to the development of artificial memory systems that share computational principles with associative memory in the mammalian brain, implemented in a neural dialog between the hippocampus and PFC.** 
    - We now understand a great deal about how biological systems learn the structure of the world (independent of its contents) in ways that allow them to deal flexibly with novelty. 
    - Surprisingly, the computational solutions identified in natural intelligence are directly analogous to those that power modern AI systems.
    - **This is what we mean by the emerging "recipe" for intelligence.**

- AI research has made enormous strides in the last decade.
    - Current systems, rather than being readily classifiable as "human-like" or even "superhuman," are instead best thought of as composed of "jagged" capabilities — some (such as their ability to prove theorems or design a website) that outclass most humans and others (such as their ability to book a flight or complete a tax return) that can, for the moment, be barely described as competent.
    - We argue that the shortcomings of current AI systems can be traced to their lack of the core functionalities that sustains natural intelligence. 
        - **For example, biological systems exchange information flexibly between short- and long-term memory, which allows them to learn continually from experience.**
        - **They model and monitor their own goals and actions, allowing them to remain true to goals over the long term.** 
        - **They learn from dynamic interactions with the environment rather than passive exposure to content, making them creative in their dealings with an open-ended world.** 
    - Future AI systems may need to incorporate these cognitive and computational principles to fulfill their promise of becoming generally intelligent assistants. 
    - There are two major ways in which neuroscience may still have something to offer AI. 
        - **The first concerns cognitive architectures — how to integrate the many component parts together into an AI system, just as brain "modules" (including subcortical regions such as the amygdala and basal ganglia) work together to generate biological intelligence.** 
        - Secondly, and not discussed here in detail, is the need for more efficient transmission and plasticity rules to bring down the computational (and energetic) cost of in silico computation to sustainable levels.
    - Of course, it is also possible that AI researchers will invent new solutions that completely ignore how biological systems work. 
    - Either way, we do not yet know the full "recipe" for intelligence — but it is definitely closer than ever before.

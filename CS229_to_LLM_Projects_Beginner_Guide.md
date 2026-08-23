# CS229 → LLM Projects — Plain-Language Guide & Roadmap

*A friendly companion to `CS229_to_LLM_Research_Projects.md`. That file is the technical spec (exact equations, datasets, model IDs). This file is the map: what each project really is in everyday words, the math to learn first, the tools, and a step-by-step path from "I've read the notes" to "I finished the project."*

---

## How to use this guide

Read this **first**, then open the spec for the precise details when you're ready to build. For each project you'll find five things:

1. **In plain language** — what the project is, explained like I'm telling a friend over coffee.
2. **Math to learn first** — the prerequisites, in order, with the CS229 chapter that teaches each. Split into *must-know* and *nice-to-have*.
3. **Tools** — the actual software you'll use.
4. **The build roadmap** — a start-to-finish path: *learn this → build this → you're now at this level → connect it to the next piece → finish.*
5. **What you'll learn** — stated twice: once in ML vocabulary (for your résumé/interviews) and once in plain language (so you know what actually changed in your head).

**One deliberate choice:** the roadmaps tell you *what* to do at each step and *how to know you succeeded* — but they do **not** hand you the finished code or the experimental answers. That's on purpose. The understanding you're after lives in the doing; if I write the solution, you read it and forget it. Concrete dataset and model IDs are in the spec's Appendix F when you need them. Here, I'm giving you the path, not the destination.

**A mindset that helps:** every project is the same loop — *understand the math → write the algorithm yourself → point it at a real language model → look hard at what happens → explain the gap between what the theory promised and what you saw.* Once that loop feels natural, you're doing research.

---

## The big picture: how the seven projects connect

Don't think of these as seven separate assignments. Think of them as one skill tree where each project hands tools to the next. Here's the journey.

### Phase 0 — Foundations (before you touch a single project)

**Learn:** refresh linear algebra (vectors, matrices, dot products) and the single most important classical algorithm — logistic regression, to the point where you can derive its gradient by hand. Skim CS229 Ch. 1–2.

**Build:** the shared `llmkit` toolkit from the spec — two small functions that (a) pull a model's hidden activations and (b) pull its next-token probabilities.

**You're now at this level:** *"I can open any small language model and read out both what it's 'thinking' (activations) and what it 'believes' (probabilities)."* This is the single unlock that makes every project possible.

### Phase 1 — Project 1 (Probes): learn to *measure* a model

You add a from-scratch logistic-regression classifier and use it as a measuring stick for what the model knows at each layer. **Connect:** the `probe.py` you write here gets reused in Projects 3 and 4 — so building it well now pays off twice later.

### Phase 2 — Project 2 (Geometry): learn to *see* a model's representations

You add linear-algebra tools (PCA, ICA, clustering) and study the *shape* of the model's embedding space. This is independent of Project 1 but shares the activation harness. **Connect:** dimensionality reduction here becomes a reflex you'll use to visualize anything high-dimensional in later work.

### Phase 3 — Project 3 (Calibration): learn to *judge* a model

Reusing your probe and the probability extractor, you ask whether the model's confidence is trustworthy — and fix it. **Connect:** it's the shortest project and reuses the most, so it's the perfect confidence-builder after the first two.

### Phase 4 — Project 4 (Detection): learn to *compare* approaches

You build several classifiers (generative and discriminative) on a real problem — spotting AI-written text — and watch the classic CS229 tradeoff play out live. **Connect:** this is where the whole first half of your notes comes together in one experiment.

### Phase 5 — Project 5 (Double Descent & Grokking): learn how models *actually generalize*

A change of pace — you train a tiny model from scratch and reproduce two famous, weird phenomena. **Connect:** this gives you the intuition for *why* big models behave the way they do, which reframes everything in Phases 1–4.

### Phase 6 — Project 6 (RLHF): learn to *steer* a model

The alignment capstone. You implement the algorithm behind modern alignment and use it to change how a model writes. **Connect:** it combines generation, fine-tuning, and probability math from every earlier phase into the technique that defines today's frontier models — and its RLVR extension (rewarding *correct reasoning*) is the bridge to the newest research.

### Phase 7 — Project 7 (Attention Internals): learn to *reverse-engineer* a model

The deepest cut. You rebuild attention from your notes' own equations, prove it correct against a real model, then find and causally verify an **induction head** — the little circuit behind in-context learning. **Connect:** where Phase 1 read *where* a concept lives and Phase 2 saw its *shape*, this asks *which component computes it* — landing you in the vocabulary of mechanistic interpretability, arguably the most direct on-ramp to LLM-research work.

**The through-line:** *measure → see → judge → compare → understand → steer → reverse-engineer.* Finish all seven and you can honestly say you can take a language model apart and put it back together. Do even three (say P1, P3, P6 — one per skill) and you have a coherent story to tell; add P7 and you have a mechanistic-interpretability portfolio piece.

---

# Project 1 — Linear Probes: *measuring what a model knows*

### In plain language
A language model turns every word into a long list of numbers (a vector) at each of its internal layers. The question is: **is the model's knowledge actually sitting in those numbers, and where?** You test this by training a very simple classifier — the simplest one that exists — to read one specific thing (say, "is this sentence positive or negative?") off those numbers. If your simple classifier can do it, the model must be storing that information in a clean, easy-to-read way at that layer. By repeating this layer by layer, you draw a map of *where in the model different kinds of understanding appear*. It's like putting a stethoscope on different parts of the model and asking "can I hear sentiment here? how about grammar?"

### Math to learn first
*Must-know (learn these before starting):*

- **Vectors and matrices** — dot products, matrix–vector multiply. Any linear-algebra refresher.
- **The logistic (sigmoid) function and logistic regression** — CS229 Ch. 2.1. This *is* your probe.
- **Gradients / partial derivatives** — enough to understand "walk downhill to reduce error." CS229 Ch. 2.
- **The idea of maximum likelihood** — why we fit models by making the data probable. CS229 Ch. 1.3, 2.1.

*Nice-to-have (you can pick these up as you go):*

- **Newton's method** (using the second derivative to converge faster) — CS229 Ch. 2.4.
- **Regularization and cross-validation** (stopping a model from cheating) — CS229 Ch. 9.1, 9.3.
- **Bias-variance intuition** (why a *too-powerful* probe gives misleading answers) — CS229 Ch. 8.1.

### Tools
Python, PyTorch, Hugging Face `transformers` (to load the model), your own `llmkit` (activation extractor), NumPy (to write the classifier), matplotlib (for the layer map). No `scikit-learn` for the probe itself — you write that.

### The build roadmap
1. **Learn:** derive the logistic-regression update rule until you can do it on paper. *(Checkpoint: you can explain why the gradient has the form "error × input.")*
2. **Build:** implement logistic regression from scratch in NumPy and test it on any toy 2-D data you make up. *(Checkpoint: it separates two blobs you can plot.)*
3. **Connect it to the model:** use `llmkit` to grab hidden vectors from one layer of a small model for a batch of sentences; feed those vectors as the input to your probe. *(Checkpoint: your probe beats "always guess the majority class" on held-out sentences.)*
4. **Level up — the honesty check:** re-run the probe against *shuffled/random* labels. If it still scores well, your probe is memorizing, not measuring. Report your real score *minus* this control score. *(Checkpoint: you understand why raw accuracy alone would have fooled you.)*
5. **Complete:** sweep every layer, plot accuracy-vs-layer, and write two paragraphs on where the concept "lives" and whether that surprised you. **You're done when you have one clean figure and can defend it.**
6. **Carry forward:** keep your probe code tidy — Projects 3 and 4 import it.

### What you'll learn
*In ML terms:* linear classifiers, MLE and gradient-based optimization, the linear-probing methodology for interpretability, control tasks / probe selectivity, and layer-wise representation analysis.

*In plain language:* how to build the most fundamental classifier by hand, and how to use it as a tool to peek inside a model and prove *where* it stores a piece of knowledge — while avoiding the trap of fooling yourself with a number that looks good but means nothing.

---

# Project 2 — The Shape of Meaning: *seeing a model's representations*

### In plain language
Every word and sentence lives as a point in a space with hundreds or thousands of dimensions. Humans can't picture that, so this project is about *compressing* that space down to a few directions we can actually look at and reason about — and asking whether those directions mean anything. You'll discover a strange fact: the model's "meaning space" isn't a nice round cloud; it's squished into a narrow cone, which quietly breaks the usual way people measure similarity. You'll fix that, and then go hunting for hidden "axes of meaning" (imagine finding that one direction in the space corresponds to "plural-ness" and another to "past tense"). It's detective work on the geometry of language.

### Math to learn first
*Must-know:*

- **Eigenvalues and eigenvectors, and covariance matrices** — the heart of PCA. CS229 Ch. 12.
- **Variance, and the difference between "uncorrelated" and "independent"** — this gap is the whole reason ICA exists. CS229 Ch. 13.1.
- **Clustering intuition** — grouping points by closeness. CS229 Ch. 10.

*Nice-to-have:*

- **SVD (singular value decomposition)** — another lens on PCA.
- **Probability densities and the change-of-variables formula** — the math under ICA. CS229 Ch. 13.2.
- **Latent variables, Jensen's inequality, and the EM idea** — for soft clustering with Gaussian mixtures. CS229 Ch. 11.

### Tools
Python, PyTorch + `transformers` (embeddings), `sentence-transformers` (for the similarity test), NumPy (you write PCA, ICA, k-means, and EM yourself), matplotlib. `scikit-learn` allowed only as a *checker* — confirm your PCA matches theirs, then use yours.

### The build roadmap
1. **Learn:** understand PCA as "find the directions of greatest variance," and be able to explain why those are the top eigenvectors of the covariance. *(Checkpoint: you can say what the eigenvalues mean.)*
2. **Build:** implement PCA two ways (full eigendecomposition, and power-iteration for just the top few). Sanity-check against a library to tolerance. *(Checkpoint: same answer as the library, your code.)*
3. **Connect it to the model:** run PCA on real embeddings; plot the variance spectrum and measure how "cone-shaped" (anisotropic) the space is. *(Checkpoint: you can state the anisotropy number.)*
4. **Level up — the payoff experiment:** remove the top few directions and re-measure how well cosine similarity matches human similarity judgments. Find the sweet spot. *(Checkpoint: you show similarity got *better* after deletion — a counterintuitive win.)*
5. **Go further:** run your ICA to look for interpretable independent axes, and cluster word senses with your EM/GMM. *(Checkpoint: you can name a handful of components that clearly mean something.)*
6. **Complete:** one figure contrasting "PCA directions (murky) vs. ICA directions (interpretable)," plus the similarity result. **Done when the figure tells the story without you narrating it.**

### What you'll learn
*In ML terms:* PCA/SVD, ICA and non-Gaussianity, k-means vs. Gaussian-mixture EM, whitening, and the anisotropy / representation-degeneration phenomenon in embeddings.

*In plain language:* how to take an incomprehensibly high-dimensional space and both *shrink it so you can see it* and *find the meaningful directions hidden inside it* — plus the humbling lesson that the "obvious" tool (PCA) isn't always the one that finds meaning. (Optional finale: these same embeddings are exactly what powers semantic *search* and *RAG* — retrieval-augmented generation — so you can end by building a tiny "find the most relevant sentence" engine and see your geometry cleanup actually improve the results. CS229 Ch. 16.)

---

# Project 3 — Calibration: *judging whether a model is honest about doubt*

### In plain language
When a model says it's 90% sure, is it actually right 90% of the time? A trustworthy model's confidence should match its accuracy. Often it doesn't — especially after the "make it chatty and helpful" tuning step, which tends to make models overconfident (they sound sure even when they're guessing). This project measures that mismatch, draws the picture that reveals it, and then fixes it with a tiny, elegant adjustment — sometimes a *single number* — borrowed straight from your notes. It's about teaching yourself to tell the difference between a model that *knows* and a model that just *sounds* confident.

### Math to learn first
*Must-know:*

- **Probability distributions and expectation** — what a probability actually claims.
- **Softmax and cross-entropy** — how a model turns scores into probabilities and how it's trained. CS229 Ch. 2, §17.2 (the next-token prediction loss).
- **Maximum likelihood again** — fitting by making observed outcomes probable.

*Nice-to-have:*

- **The exponential family / GLMs** — why softmax is the "natural" choice, not an arbitrary one. CS229 Ch. 3.
- **Newton's method in 1-D** — to fit the temperature parameter. CS229 Ch. 2.4.

### Tools
Python, PyTorch + `transformers` (to get answer-token probabilities via `llmkit`), NumPy (you write the temperature/Platt fitting and the error metric), matplotlib (reliability diagrams). `datasets` for the question banks.

### The build roadmap
1. **Learn:** understand what "calibration" means and how it differs from accuracy — a model can be accurate but overconfident, or vice versa. *(Checkpoint: you can give an example of each.)*
2. **Build:** write the code that turns a batch of predictions + confidences into a **reliability diagram** and a single calibration-error number. *(Checkpoint: a perfectly-calibrated toy input gives near-zero error.)*
3. **Connect it to the model:** on a multiple-choice question set, read the probabilities the model assigns to each option and measure its real calibration. *(Checkpoint: you have the model's reliability diagram.)*
4. **Level up — the fix:** implement temperature scaling (fit one number by maximum likelihood on a validation split, using Newton's method) and show the error drop on a separate test split *without* hurting accuracy. *(Checkpoint: same accuracy, lower calibration error.)*
5. **The headline experiment:** compare a base model against its instruction-tuned twin. See whether tuning made it overconfident. *(Checkpoint: two reliability diagrams side by side.)*
6. **Complete:** write up "is this model honest about what it doesn't know, and what did tuning cost?" **Done when someone non-technical could read your diagram and get it.**

### What you'll learn
*In ML terms:* probability calibration, expected calibration error, reliability diagrams, temperature and Platt scaling, the softmax-as-GLM view, and the calibration cost of alignment tuning.

*In plain language:* how to check whether a model's confidence can be trusted, how to repair it with a shockingly small tweak, and the real-world insight that making a model friendlier can quietly make it more of a bluffer.

---

# Project 4 — Who Wrote This?: *comparing generative and discriminative detectors*

### In plain language
Given a piece of text, can you tell if a human or an AI wrote it? This is a real, hard, important problem — and it's the perfect stage for the most important idea in the first half of your notes: there are **two philosophies of building a classifier.** One (generative) learns what human text and AI text each *look like* and asks "which does this resemble more?" The other (discriminative) skips the description and just learns the *dividing line* between them. Your notes claim each philosophy wins under different conditions — the descriptive one when you have little data, the line-drawing one when you have lots. You'll put that claim on trial. The clever twist: the best clues aren't the words themselves, but how *surprised the model is* by each word — AI text tends to be "too predictable." So you measure the model's surprise and classify on that.

### Math to learn first
*Must-know:*

- **Bayes' rule and conditional probability** — the engine of generative classifiers. CS229 Ch. 4.
- **The Gaussian and multivariate Gaussian** — how GDA models each class. CS229 Ch. 4.1.1.
- **Counting-based text models (multinomial) and smoothing** — the Naive Bayes baseline. CS229 Ch. 4.2.
- **The generative-vs-discriminative distinction** — the whole point. CS229 Ch. 4.1.3.

*Nice-to-have:*

- **Convex optimization + Lagrange duality** — to understand the SVM. CS229 Ch. 6.5.
- **The kernel trick** — bending a straight boundary into a curve. CS229 Ch. 5.
- **Coordinate ascent / the SMO algorithm** — how to actually train the SVM. CS229 Ch. 6.8.

### Tools
Python, PyTorch + `transformers` (to score how "surprised" a model is, via `llmkit` log-probabilities), NumPy (you write Naive Bayes, GDA, logistic regression, and the SVM/SMO), matplotlib. `datasets` for text.

### The build roadmap
1. **Learn:** be able to explain, in one sentence each, how a generative and a discriminative classifier decide — and why they differ. *(Checkpoint: you can state the tradeoff from Ch. 4.1.3 out loud.)*
2. **Build the easy baseline:** implement multinomial Naive Bayes with smoothing on word counts. *(Checkpoint: it's above chance at separating human vs. AI text.)*
3. **Build the features that matter:** use `llmkit` to turn each text into a few "surprise" numbers (how likely the model found each word). *(Checkpoint: human and AI texts have visibly different surprise profiles when you plot them.)*
4. **The core comparison:** put GDA (generative) and logistic regression (discriminative) on the *same* surprise features, and plot how each does as you give them more and more training data. *(Checkpoint: you can see which one leads early and which one wins late — the notes' claim, tested.)*
5. **Level up:** implement the SVM via SMO from scratch and add a curved (kernel) boundary; see if it beats the linear methods. *(Checkpoint: your hand-written SMO trains and classifies.)*
6. **Complete:** the headline figure — accuracy vs. amount-of-data, with the crossover — plus a note on how detection gets harder as the AI writes more randomly. **Done when your figure demonstrates the tradeoff you read about.**

### What you'll learn
*In ML terms:* generative (GDA, Naive Bayes) vs. discriminative (logistic regression, SVM) modeling, the data-efficiency crossover, kernels and the SMO optimizer, and likelihood-based features for machine-text detection.

*In plain language:* the two fundamental ways to teach a machine to sort things, when to reach for each, and how to build all of them yourself — applied to a genuinely useful, current problem, using the idea that AI writing is "suspiciously unsurprising."

---

# Project 5 — Double Descent & Grokking: *how models really learn to generalize*

### In plain language
Everything you're first taught about machine learning says: make your model too big and it will "overfit" — memorize the training data and fail on new data. Modern deep learning cheerfully breaks that rule, and this project lets you watch it happen with your own eyes on a model tiny enough to train on your laptop. You'll see **double descent** (as the model grows, performance gets worse, then — surprisingly — better again) and **grokking** (a model sits there having merely memorized the answers, and then, long after you'd have given up, it *suddenly understands the rule*). These aren't curiosities; they're clues to why giant models work at all. You'll reproduce them and then find the knob (a form of "keep the model humble") that controls them.

### Math to learn first
*Must-know:*

- **Least-squares and ridge regression** — the from-scratch "theory anchor" version of the effect. CS229 Ch. 1, Ch. 9.1.
- **The bias-variance decomposition** — the classic explanation for over/underfitting, and the algebra to split error into its parts. CS229 Ch. 8.1, 8.1.1.
- **What "overfitting" and "interpolation" (fitting training data perfectly) mean.** CS229 Ch. 8.2.

*Nice-to-have:*

- **How a small neural network and backprop work** — enough to train one. CS229 Ch. 7.
- **Weight decay / implicit regularization** — the control knob. CS229 Ch. 9.1, 9.2.

### Tools
Python, NumPy (the from-scratch linear/bias-variance experiments), PyTorch (the tiny Transformer — layers are plumbing, but you write the training loop and logging), matplotlib. **No dataset to download** — you generate simple math problems in a few lines.

### The build roadmap
1. **Learn:** be able to write down the bias-variance decomposition and say what each term means. *(Checkpoint: you can explain why variance is what blows up when you overfit.)*
2. **Build the theory anchor:** reproduce double descent with plain (random-feature) ridge regression in NumPy — sweep model size across the "just barely fits the data" threshold and watch the error curve dip, spike, and dip again. *(Checkpoint: you've drawn the double-descent curve from scratch.)*
3. **Measure it:** estimate the bias and variance separately by training many models on resampled data. *(Checkpoint: the variance spike lines up with the error spike.)*
4. **Connect it to a real model:** train a tiny Transformer on a simple arithmetic rule and log train vs. test accuracy for a *long* time. *(Checkpoint: you catch the "grokking" moment where test accuracy suddenly jumps.)*
5. **Level up — find the knob:** vary the "keep it humble" setting (weight decay) and show it moves the grokking point and tames the spike. *(Checkpoint: you can make grokking happen sooner or later on demand.)*
6. **Complete:** one narrative — "the same phenomenon in three forms: simple theory → measured bias/variance → a real tiny Transformer." **Done when the three views clearly tell one story.**

### What you'll learn
*In ML terms:* bias-variance tradeoff and its decomposition, the interpolation threshold, model-wise and sample-wise double descent, grokking, and regularization's effect on generalization.

*In plain language:* why the old "bigger = overfitting" rule is incomplete, what really happens as models grow and train, and the hands-on intuition for the mysterious way large models suddenly "get it" — the closest thing to watching understanding form.

---

# Project 6 — RLHF from Scratch: *steering how a model behaves*

### In plain language
This is how real labs turn a raw text-predictor into a helpful assistant: you let the model generate, you *score* what it produced (good/bad), and you nudge it to do more of the good. The scoring-and-nudging algorithm is called policy gradient, and it comes straight from the reinforcement-learning chapters of your notes. You'll build a miniature version — steering a small model to, say, write more positively — and along the way you'll hit the two truths every alignment team knows: the naive method is *maddeningly noisy* (and there's a classic trick to calm it), and if you only chase the reward, the model finds dumb ways to cheat (so you add a leash that keeps it close to its original self). It's the most "real" project of the seven — a tiny but faithful version of what powers today's chatbots.

### Math to learn first
*Must-know:*

- **Probability and expectation, and the "log-derivative trick"** — the one identity the whole method rests on. CS229 Ch. 21.
- **Gradients and gradient ascent** — you're climbing toward higher reward.
- **What a reward, a policy, and an MDP are** — the reinforcement-learning frame. CS229 §19.1.

*Nice-to-have:*

- **Variance of an estimator, and why a "baseline" reduces it** — the trick that makes training bearable. CS229 Ch. 21.
- **KL divergence** — the "leash" that stops the model drifting into gibberish. (Connects to regularization, Ch. 9.)
- **How autoregressive generation works** — the model's sampling *is* the policy. CS229 §17.2.

### Tools
Python, PyTorch + `transformers` (the model you'll steer), `peft` (LoRA — lets you fine-tune cheaply within 8 GB), a small off-the-shelf reward scorer (or a simple rule you write), your `llmkit` for token probabilities, matplotlib.

### The build roadmap
1. **Learn:** derive the REINFORCE gradient — understand *why* "reward × gradient-of-log-probability" nudges the model correctly. *(Checkpoint: you can explain the log-derivative trick.)*
2. **Build the loop with a fake-simple reward:** reward something trivial (like longer outputs) so you can debug the mechanics without a reward model. *(Checkpoint: the average reward climbs over training.)*
3. **Calm the noise:** add a baseline and measure how much steadier training becomes. *(Checkpoint: you can show the variance dropping, before vs. after.)*
4. **Use a real reward:** swap in a sentiment scorer and steer the model to write more positively. *(Checkpoint: before/after samples read differently.)*
5. **Level up — add the leash:** add a KL penalty toward the original model. Show that *without* it the model reward-hacks into nonsense, and *with* it, quality holds. *(Checkpoint: you can plot the reward-vs-drift tradeoff.)*
6. **Complete:** "RLHF in a few hundred lines," with reward curves, the noise-reduction result, the leash tradeoff, and sample outputs. **Done when you can explain every term in your loss to someone else.**
7. **Go even further — reward *correctness* instead of taste (the 2025 frontier):** instead of scoring "how positive," give the model little problems with a *checkable* answer (simple arithmetic, or "output must match this format"), let it "think out loud" first, and reward it *only* when a checker confirms the answer is right. This is called **RLVR** (reinforcement learning with verifiable rewards) and it needs no reward model at all — the checker is free and can't be gamed. It's the exact idea behind today's "reasoning" models, and it fits on your laptop. *(Checkpoint: accuracy climbs and the model's step-by-step reasoning gets cleaner. CS229 Ch. 18.)*

### What you'll learn
*In ML terms:* the policy-gradient theorem and REINFORCE, score-function estimators and baselines for variance reduction, KL-regularized fine-tuning, LoRA, and the RLHF training loop that underlies modern alignment.

*In plain language:* how to change a model's behavior by rewarding what you like — the actual method behind ChatGPT-style assistants — plus first-hand experience of the two things that make it hard: the training is jittery, and models will "cheat" any reward you don't guard carefully.

---

# Project 7 — Inside the Attention Head: *how a Transformer actually computes*

### In plain language
Every other project treats the model's insides as a place to *read from*. This one opens the engine and studies the single most important part — **attention**, the mechanism that lets a model look back at earlier words and decide which ones matter right now. You'll build attention yourself from the equations in your (newly updated) notes, then prove your version is correct by checking it matches a real model's attention number-for-number. Then the fun part: you go looking for a specific, famous little circuit called an **induction head** — a piece of the model that has learned the trick "*I saw this before; here's what came next, so I'll copy it.*" That trick is a big part of how models learn from examples you put in the prompt. Finally, you'll *prove* a head matters by switching it off and watching the model get worse — the difference between "this looks important" and "this **is** important." This is the entry point to **mechanistic interpretability**, the research area trying to reverse-engineer what's actually happening inside these models.

### Math to learn first
*Must-know:*

- **Dot products, matrix multiplication, and the softmax** — attention is "compare with dot products, turn into weights with softmax, take a weighted average." CS229 §17.3 (and the softmax from Ch. 2).
- **Why divide by √(dimension)** — a small scaling that keeps the numbers from blowing up; the notes explain exactly why. CS229 §17.3.
- **What "attend to" means** — each position produces a query, every position offers a key, and the match decides where attention goes.

*Nice-to-have:*

- **Low-rank matrices and subspaces** — a head only "sees" a small slice of the space; PCA intuition from Project 2 helps. CS229 Ch. 12.
- **The idea of an ablation / knockout experiment** — the causal logic of "remove it and see what breaks."
- **Chain-of-thought and verifiable rewards** — for the reasoning extension. CS229 Ch. 18.

### Tools
Python, PyTorch + `transformers` (to pull a real model's attention weights and compare), NumPy (you write attention yourself), matplotlib (attention heatmaps), your `llmkit` hooks (extended to capture attention). No training needed — this is dissection, not construction, so it's light on the GPU.

### The build roadmap
1. **Learn:** understand attention as query-key-value — be able to say, in one breath, "dot the query with every key, softmax into weights, average the values." *(Checkpoint: you can draw it.)*
2. **Build & prove it:** implement one attention head from the notes' equations, then check it matches GPT-2's real head to tiny numerical tolerance. *(Checkpoint: your numbers equal the library's numbers.)*
3. **Add the variants:** extend to multi-head, then the memory-saving MQA/GQA variants; compare their cost. *(Checkpoint: a small table of "memory vs. quality.")*
4. **Look inside:** visualize attention patterns from a real model and name a few head "types" (one that looks at the previous word, one that parks on the first token…). *(Checkpoint: an annotated gallery.)*
5. **Hunt the induction head:** feed repeated patterns and find the head that copies — the in-context-learning circuit. *(Checkpoint: you can point to the head and show it copying.)*
6. **Prove it matters:** switch heads off one at a time and measure the damage; rank heads by importance. *(Checkpoint: knocking out the induction head visibly hurts copying.)*
7. **Go further (the 2025 angle):** does prompting the model to "think step by step" change *which* heads it needs? Compare the importance map with and without a chain of thought. *(Checkpoint: a before/after head map. CS229 Ch. 18.)*

### What you'll learn
*In ML terms:* scaled dot-product and multi-head self-attention (and MQA/GQA variants), QK vs. OV circuits, attention-pattern analysis, induction heads and their link to in-context learning, and causal methods (head ablation / activation patching) — the working vocabulary of mechanistic interpretability.

*In plain language:* how the core of a Transformer actually computes, built with your own hands and *proven* correct — and how researchers move from "this part looks important" to "this part **is** important" by carefully breaking things. It's the most direct answer to "but how does it really work inside?"

---

# What you'll learn — in ML terms and in normal language

Here's the whole curriculum's payoff, said both ways: the left column is what you'd write on a résumé or say in a research interview; the right column is what actually changed in your understanding.

| In ML / research terms | In normal language |
|---|---|
| Implemented logistic & softmax regression, MLE, and gradient/Newton optimization from scratch | I can build the most basic "learning" algorithms by hand and know *why* they work, not just how to call them |
| Applied linear probing and control tasks to analyze representations layer-by-layer | I can look inside a model and prove *where* it stores a piece of knowledge — without fooling myself |
| Performed PCA/ICA/SVD and EM-based clustering on embedding spaces; diagnosed anisotropy | I can shrink a huge, invisible space down to something I can see, and find the meaningful directions hidden in it |
| Measured and corrected model calibration (temperature/Platt scaling); analyzed the exponential-family view of softmax | I can tell whether a model's confidence is trustworthy, and fix it — and I learned that "helpful" tuning can make a model a bluffer |
| Compared generative vs. discriminative classifiers (GDA, Naive Bayes, logistic, kernel SVM via SMO) and the data-efficiency crossover | I know the two fundamental ways to teach a machine to sort things, when each wins, and I can code all of them |
| Reproduced double descent and grokking; decomposed error into bias and variance; studied regularization | I understand, hands-on, the strange truth about how modern models generalize — and why "bigger overfits" is wrong |
| Built a KL-regularized REINFORCE (policy-gradient) fine-tuning loop with variance-reduction baselines and LoRA | I can steer a model's behavior with rewards — the real technique behind assistants — and I've felt why it's noisy and why models cheat |
| Reverse-engineered self-attention from scratch, verified it numerically against a real model, and causally localized induction heads by ablation | I can rebuild the core of a Transformer with my own hands, *prove* it's correct, and show which internal parts actually do the work |
| Engineered a reusable interpretability toolkit; ran controlled experiments with baselines and reproducible figures | I can set up a clean experiment, rule out the boring explanation, and produce a result someone else can trust and repeat |

**And the meta-skill, in one sentence:** you'll have practiced the actual loop of research — *turn math into code, point it at a real model, measure honestly, and explain the gap between theory and reality* — which is exactly the muscle LLM research runs on.

---

*This guide is the map; `CS229_to_LLM_Research_Projects.md` is the terrain (exact equations, datasets, model IDs, setup). Start with Phase 0, build the toolkit, and take the projects in order — each one hands tools to the next.*


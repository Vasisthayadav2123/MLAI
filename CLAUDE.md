# ML Learning Project — Tutor Instructions

## Role
Act as my Machine Learning tutor and coding mentor. I'm learning ML from scratch and want to understand the concepts and math deeply before I use libraries.

- **Do NOT write my ML implementation code.** Explain concepts, derive the math, give me exercises, review my code, and give hints one level at a time when I'm stuck.
- Point out bugs as questions or hints, not fixes. Small illustrative math or pseudocode is fine; full working solutions are not.
- Style is **mix**: explain genuinely new ideas fully, but make me derive anything that builds on what I already know.
- Move to libraries (NumPy, scikit-learn, PyTorch) only after my from-scratch version works.

## About me
- Calculus: basic. I know simple derivatives; the chain rule and partial derivatives were shaky (as of 2026-09-29). As of 2026-10-25 I can apply a nested chain rule with step-by-step hints; watch for dropped minus signs.
- Goal: get an ML/AI job, so prioritize portfolio-worthy projects and industry tools (after I understand the internals).

## Roadmap
1. **Neural network core:** chain rule → finish logistic regression (BCE loss + gradient) → single neuron → 2-layer NN with hand-derived backprop. Pure Python first (XOR) → **project: rockets learning to fly** (neuroevolution/genetic algorithm: each rocket has a small NN brain, best flyers are copied with mutations each generation; visual demo, portfolio piece) → then NumPy (MNIST from scratch).
2. **Classical ML:** train/test split, overfitting, regularization, kNN, decision trees, Naive Bayes, k-means, PCA → then scikit-learn. Tabular data project.
3. **Deep learning:** mini autograd engine, optimizers (momentum/Adam), CNNs, RNNs → attention/Transformers → then PyTorch. Image classifier + small character-level language model.

## Progress
- Done (pure Python): vector/matrix ops (`core_python.py`), 1D linear regression (`linear_regression_rawdogg.py`), multi-feature linear regression (`_linear_regression.py`), animated gradient-descent visualization (`visualise_Linear_regression.py`), logistic regression stub with sigmoid only (`logistic_regression.py`).
- ✅ **Lesson 1 (2026-09-29 → 2026-10-25): the chain rule.** Exercises:
  1. ✅ `gradient_check.py`: numerical gradient check against my `gradients()` function (done 2026-10-24; passes ~1e-10, caught a deliberately removed factor of 2 in db). Still shaky on why central difference divides by 2h.
  2. ✅ Derive σ'(z) = σ(z)(1 − σ(z)) on paper (derived 2026-10-25 with step-by-step hints; initially missed the minus sign from d/dz e^-z), verified numerically in `logistic_regression.py` (~1e-11). Understands max σ' = 0.25 and gradient shrinks when confident.
  3. ✅ Explain why the ~1200-sized feature forces learning_rate = 1e-7 (the unused `standardize_data`). Initially thought large w caused large gradient; corrected to large feature value x. Understands divergence → nan and why standardizing lets one learning rate work. (Optional, not yet done: actually train on standardized features.)
- ✅ **Lesson 2 (2026-10-26): binary cross-entropy loss.**
  - ✅ 2.1 Derived MSE+sigmoid gradient 2(p−y)p(1−p); showed confidently-wrong model (p=0.99, y=0) gets 15× smaller gradient than p=0.6; identified p(1−p) as the culprit.
  - ✅ 2.2 Derived BCE dL/dz = p − y case by case (y=1, y=0) with hints. Got confused about where z is in link 2 (forgot p = σ(z)); needs brackets in fractions like −1/(1−p).
  - ✅ 2.3 dL/dw_j = (p−y)x_j, dL/db = p−y. First gave only the dz/dw link (forgot to multiply by dL/dz); needed to be shown how to reuse the boxed p − y. Spotted the missing factor 2 vs linear regression.
  - ✅ 2.4 Coded logistic regression in `logistic_regression.py` (hours-studied toy data): BCE with log(0) clipping, averaged gradients, training (lr 0.1, 10k epochs → loss 0.3129, w=1.28, b=−5.77, boundary 4.5h), accuracy 75%. Skipped the gradient check by choice. Bugs hit: forgot to average gradients; passed predictions into bce_loss (copied mse call signature). Understands non-separable data → loss can't reach 0, and that a single neuron is monotonic.
  - ✅ 2.5 Found threshold 3.5 gives 87.5%; understands gradient descent minimizes loss not accuracy, and accuracy has zero gradient (surrogate loss idea).
- ✅ **Lesson 3 (2026-10-27): single neuron, decision boundaries, why one neuron can't solve XOR.**
  - ✅ 3.1 By hand: AND (w=1,1 b=−1.5) correct; OR needed a hint (tried b=0, didn't see b in (−1,0) works); XOR impossibility proved via inequalities with heavy guidance (struggled with "positive + positive > 0" step and read `1·0` as decimal 1.0; use × for multiplication). Algebra with inequalities is a weak spot.
  - ✅ 3.2 `neuron.py` (2-feature neuron, imports sigmoid/dot_product): AND & OR 100%, XOR stuck at w=[0,0], b=0, loss ln2=0.693, 50%. Explained it as symmetry (gradient exactly 0 at zero init). Told: best single line gets 75% on XOR, but ln2 is the convex loss minimum; zero-init symmetry is a trap in multi-neuron nets → random init. Pending small fix: main loop not indented under `if __name__`. Skipped: XOR loss prediction, per-point p printout, boundary plot bonus.
- ✅ **Lesson 4 (2026-10-27 → 2026-10-29): 2-layer NN with hand-derived backprop** (solve XOR; random init; ReLU/tanh).
  - ✅ 4.1 Hand-designed XOR = OR and not AND (hidden OR/AND neurons, output v=[1,−1], c=−0.5). Got the h-table and "h1 high, h2 low"; flipped signs twice when moving terms across inequalities (c range, v2 < −0.5).
  - ✅ 4.2 `two_layer.py` forward pass with ×10 weights → p ≈ 0.007, 0.992, 0.992, 0.007. Needed 3 rounds: first built one hidden neuron, then chained neurons in series; fixed via fill-in-the-blank skeleton. Hidden-layer = parallel neurons on the same x was the confusing part.
  - ✅ 4.3 Output layer grads (p−y)·a1, (p−y)·a2, p−y by substitution (asked "what is L?"; needed a worked row before filling in).
  - ✅ 4.4 Derived ∂L/∂w1[0] = (p−y)·w3[0]·a1(1−a1)·x[0] cleanly from a 4-link scaffold. Introduced delta notation (δ3 = p−y, δ1 = δ3·w3[0]·a1(1−a1); grad = δ × input).
  - ✅ 4.5 δ2 (first used w3[0], fixed to w3[1]); grumbled "why so many" gradients, got the δ × input pattern. Coded backprop + training in `two_layer.py` (params as flat list of 9, random.seed(0), lr 1, 10k epochs) → loss 0.00098, XOR solved; hidden units rediscovered OR/AND. Bugs hit: used MSE delta 2(p−y)p(1−p) instead of BCE p−y; forward_pass signature mismatch (read global params). Skipped gradient check by choice.
  - Optional experiments offered: zero init (symmetry trap), different seeds. Not covered yet: ReLU/tanh (do with NumPy in Lesson 7).
  - Summarized hidden layers in own words ("each neuron = one line; output neuron combines them into a new feature"). Refined: output's line lives in (a1, a2) space. Open question asked: can XOR be solved with no hidden-layer activation? (Answer: no, linear layers collapse into one line.)
- **Next: Lesson 5: genetic algorithms** (fitness, selection, mutation, crossover; simple physics) → Lesson 6 rockets project (reuses `forward_pass`, no backprop needed).

# Physics-Informed Performer (PIP)
## Overview

The Physics-Informed Performer (PIP) is a transformer-based framework designed to automate the symbolic computation of squared amplitudes in high-energy physics (HEP), mapping amplitudes to their squared forms using the Task 1.2 dataset (~15K records). Unlike traditional transformers, PIP integrates the Performer’s efficient kernel-based attention with a physics-informed tokenization scheme (inspired by the Physics Informed Token Transformer, PITT) and a SymPy validation layer. This approach emphasizes novel data representation and preprocessing (Task 3.6), offering a lightweight, physics-aware alternative to symbolic manipulation for HEP cross-section predictions.

## Task 1: Data Extraction and Preprocessing
The project begins with preprocessing the Task 1.2 dataset, provided in raw text format (e.g., "-1/2*i*e^2*gamma_{+%\sigma_165,...}"). Using Python’s re module, I extract amplitude and squared amplitude pairs, converting them into a structured CSV format. Key challenges include normalizing inconsistent indices (e.g., %sigma_165 → INDEX_0) and momentum terms (e.g., p_1 → MOMENTUM_0), ensuring uniformity across records. Additional metadata (e.g., “event type,” “Feynman diagram”) is preserved to inform tokenization, enhancing the dataset’s physics context.

## Task 2: Tokenization
PIP employs a Physics-Informed Tokenization method, inspired by PITT’s structural approach for PDEs, adapted for HEP expressions. Unlike standard methods (e.g., BPE, WordPiece) that split terms like s_13 into s, _, 13, losing semantic meaning, this tokenizer preserves HEP-specific symbols as atomic units:

`<SQUARE>` for squaring operations (e.g., |A|^2).
`<S>`, `<T>`, `<U>` for Mandelstam variables (e.g., s_13 → `<S>`_INDEX_0).
`<GAMMA>` for Dirac matrices (e.g., gamma_{INDEX_0,INDEX_1,INDEX_2}).
`<QED>` for process context.
This method, detailed in the PITT paper (arXiv:2305.19192), outperforms generic tokenizers by embedding domain knowledge, improving accuracy for symbolic tasks.

## Task 3: Transformer Baseline
A vanilla transformer (Task 2) was trained on the tokenized dataset as a baseline, achieving reasonable sequence accuracy (~90% after 10 epochs). However, it struggled with long sequences (up to 509 tokens) and symbolic equivalence, prompting the development of PIP for Task 3.6.

## Task 3.6: Approach - Physics-Informed Performer (PIP)
Why This Approach?
PIP addresses the limitations of standard transformers in HEP by combining efficiency, physics awareness, and empirical validation. It replaces the computationally heavy SymKAN-TP-Transformer with a scalable, data-driven solution, leveraging Performer’s linear attention for long-sequence efficiency.Physics-informed tokens for HEP-specific reasoning.SymPy for symbolic correctness.
## How It Works
**Performer Transformer
Purpose:** Efficiently models long sequences (e.g., 40+ tokens) from Task 1.2 data.
Mechanism: Uses Favor+ attention (O(n) complexity vs. O(n²) in standard transformers), approximating attention with random feature maps. This captures token relationships (e.g., `<S>`_INDEX_0 to `<PRODUCT>`) without explicit tensor products.
Architecture: Encoder-decoder stack (4 layers each), with 512-dimensional embeddings, 8 attention heads, and feed-forward networks (2048 hidden units).
Why Chosen: Scales to HEP’s complex expressions, reducing memory and compute demands compared to TP-Transformer.

## Physics-Informed Tokenization
Preprocesses inputs with tokens like `<SQUARE>`, `<S>`, and `<GAMMA>`, embedding HEP kinematics and operations.
Benefit: Guides the Performer to focus on physics-relevant patterns (e.g., Mandelstam terms), enhancing symbolic mapping accuracy.
Example: "-1/2*i*e^2*gamma_{+%\sigma_165,...}" → [`<QED>`, -1/2, *, i, *, e^2, `<PRODUCT>`, `<GAMMA>`_INDEX_0_INDEX_1_INDEX_2, ...].

## SymPy Validation Layer
Post-processes Performer outputs to ensure empirical consistency.
Process: Detokenizes predictions (e.g., 2*e^4*(m_e^4 + ...)) into SymPy expressions, checks symmetry (e.g., |A|^2 is real), units (e.g., GeV²), and equivalence to ground truth.
Advantage: Robustly verifies mathematical structure beyond token matching, critical for HEP.

## Model Efficiency
Complexity: Performer’s O(n) attention reduces the bottleneck of standard transformers (O(n²)), making PIP viable on modest hardware (e.g., 8GB GPU).
Trade-Off: Slightly higher tokenization overhead due to physics-specific preprocessing, balanced by Performer’s lightweight attention.

## How PIP Works
Input: Raw amplitude (e.g., "-1/2*i*e^2*gamma_{+%\sigma_165,...}") is tokenized into a physics-informed sequence.

Encoding: Performer encoder processes the sequence with Favor+ attention, producing contextual embeddings.

Decoding: Performer decoder generates the squared amplitude sequence autoregressively (e.g., [<QED>, 2, *, e^4, ...]).

Validation: SymPy layer converts tokens to expressions, validates correctness, and adjusts if needed (e.g., via beam search).

Output: Final squared amplitude (e.g., "2*e^4*(m_e^4 + ...)").
Project Status

PIP is under development as part of a 175-hour GSoC project. Current progress includes tokenization implementation and baseline training, with ongoing work on Performer integration and SymPy validation.
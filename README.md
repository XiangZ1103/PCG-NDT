# PCG-NDT

Physics-Concept-Guided Neural Decision Tree for radar jamming recognition.

The model combines task-specific physical concepts with neural features in a
three-node probabilistic tree. The four output classes are narrowband noise
jamming (NNJ), interrupted-sampling repeater jamming (ISRJ), dense false-target
jamming (DFTJ), and compound jamming (CJ).

## Architecture

| Component | Configuration |
|---|---|
| DTFE | Four encoders; stage widths 48, 96, 192, 384; 512-D output |
| Node1 concepts | Six descriptors of narrowband prominence, concentration, persistence, and directionality |
| Node1 weak-ridge branch | Horizontal filters with lengths 9, 17, 33, 65; 8-D pooling descriptor; 128-D embedding |
| Node1 correction | A 10-D confidence context controls a one-sided increase in the root logit |
| Node2 | Five concepts; 672-D neural fusion; four-dimensional physical consistency; Noisy-OR probability fusion |
| Node3 | Three concepts; 672-D neural fusion; physical logit with a bounded neural residual |

The complete default model contains 5,363,575 trainable parameters (about
5.364 million): 2,551,958 in Node1, 1,405,900 in Node2, and 1,405,717 in Node3.

DTFE denotes the Decoupled Time-Frequency Encoder. It uses a parallel axial stem,
axis-separable residual blocks, and dual-axis statistics pooling. Normalization
within the encoders uses GroupNorm.

The class order of `class_probability` is:

```text
index 0: NNJ  = p1 * (1 - p2)
index 1: ISRJ = (1 - p1) * p3
index 2: DFTJ = (1 - p1) * (1 - p3)
index 3: CJ   = p1 * p2
```

## Installation

An existing environment with the dependencies already installed can run the
source directly from the repository root. No package installation is needed
for the examples and tests below.

For a new Python 3.11 environment, install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The reference dependencies specify PyTorch 2.2.2, the version reported in the
paper. The publication source was also checked in an existing environment
with Python 3.11.15, PyTorch 2.7.1+cu118, NumPy 2.4.3, and SciPy 1.17.1.
A matching CUDA-enabled PyTorch build can be used for GPU execution.

## Inspect the model

From the repository root:

```bash
python -m examples.inspect_model
```

Or inspect it from Python:

```python
from pcg_ndt import build_model, architecture_summary

model = build_model()
print(architecture_summary(model))
```

`build_model()` creates an untrained network. Its default identity concept
scalers and neutral consistency initialization are provided for architecture
inspection. Evaluation requires trained parameters and concept scalers fitted
on simulated training data.

## Inputs and inference

`prepare_inputs` accepts three sample-aligned arrays:

- `stft_db`: STFT magnitude in decibels, shape `(N, 256, 368)`;
- `b_pulses`: complex matched-filter pulse-compression outputs;
- `c_pulses`: complex IQ sequences.

The paper uses 12,000 IQ samples per example. The method extracts the 6/5/3
physical concepts, normalizes the STFT, and constructs the temporal,
pulse-compression, and IQ profiles. Concept standardization uses the scalers
stored in the model.

```python
import torch
from pcg_ndt import load_bundle

model = load_bundle("checkpoint.pt", device="cpu")
prepared = model.prepare_inputs(stft_db, b_pulses, c_pulses)
batch = prepared.tensor_batch(0, len(prepared), model.device)

with torch.inference_mode():
    output = model(batch)

probabilities = output["class_probability"]
predictions = output["prediction"]
```

By default, neural input arrays are quantized to float16 and returned as
float32. Set `quantize_inputs=False` when float32 preprocessing is required.
Physical concept arrays retain float32 precision.

## Checkpoints

`save_bundle(model, path)` stores model settings, learned parameters, and
concept scalers together. `load_bundle(path)` restores the complete state with
strict tensor-name and shape checks.

The source package contains model and preprocessing implementations. It does
not include datasets, training scripts, experiment logs, or pretrained
checkpoints. The reported experimental accuracy requires the corresponding
trained parameters and evaluation data.

Training on all-constant spectrograms produced non-finite encoder gradients
in the validation environment. Check training inputs for these degenerate
samples. The inference checks include a constant spectrogram.

## Tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

## Files

```text
pcg_ndt/
    dtfe.py       Decoupled Time-Frequency Encoder
    features.py   Physical concepts and signal profiles
    node1.py      Root decision and weak-ridge enhancement
    nodes.py      Narrowband and non-narrowband decisions
    prepared.py   Input validation and tensor batching
    model.py      Three-node model, soft routing, and checkpoints
examples/
    inspect_model.py
```

See `MODEL_STRUCTURE.md` for the input contracts and decision equations.

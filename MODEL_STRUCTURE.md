# Model structure

## Inputs

The normalized STFT has shape `(N, 1, 256, 368)`. Physical concepts are computed
from the STFT in decibels, complex IQ samples, and matched-filter outputs.
Node1, Node2, and Node3 use six, five, and three physical concepts respectively.

## DTFE

Each of the four DTFE instances has:

1. Parallel 7-by-1 and 1-by-7 convolutions with 16 output channels per path.
2. Concatenation, GroupNorm, SiLU, and max pooling.
3. Four stages of two axis-separable residual blocks, with widths
   `(48, 96, 192, 384)` and temporal dilations `(1, 2)` in each stage.
4. Per-channel global mean, frequency-marginal maximum, temporal-marginal
   maximum, and temporal-marginal standard deviation.
5. A linear projection, LayerNorm, and SiLU producing 512-dimensional features.

## Node1

Node1 estimates membership in the narrowband group `{NNJ, CJ}`.

Its physical descriptors are:

| Concept | Implementation name |
|---|---|
| Median ridge prominence | `H_ridge_prom_med` |
| Lower-quartile ridge prominence | `H_ridge_prom_q25` |
| Normalized half-prominence width | `H_ridge_bandwidth_ratio` |
| Multiscale contrast strength | `H_ms_zmax` |
| Temporal center consistency | `H_ms_center_consistency` |
| Directional response ratio | `O_anisotropic_response_ratio` |

The base branch combines a DTFE representation with the physical probability.
The weak-ridge branch uses a second DTFE and four horizontal ridge filters,
with lengths `(9, 17, 33, 65)`. Frequency attention and maximum pooling produce
an eight-dimensional descriptor, mapped to a 128-dimensional ridge embedding.

The correction is:

```text
delta_res = ReLU(logit(p_WR) - logit(p_base))
E_res = delta_res * p_ridge ** alpha
tau = 0.05 + softplus(threshold_net(context_10D))
gate = sigmoid((E_res - tau) / T_gate)
logit(p1) = logit(p_base) + gate * E_res
```

The confidence context contains the base/weak-ridge/physical logits and
uncertainties, the positive logit difference, ridge evidence, ridge logit,
and feature gate. Logits and probabilities are clipped for numerical stability.

## Node2

Node2 estimates `CJ` conditional on the narrowband group. It uses five concepts:
peak temporal z-score, temporal peak above background, mean of the largest 5%
of temporal z-scores, sustained-response duration, and secondary envelope
autocorrelation peak.

The neural feature consists of 512 STFT, 64 temporal-profile, 64
pulse-profile, and 32 direct-concept features, giving 672 dimensions. A
640-to-128-to-5 auxiliary head predicts concepts for training supervision.

```text
s_strong = min((c21 - 10) / 2, (c22 - 3) / 0.5)
s_persistent = min(c23 - 4, (c24 - 0.020) / 0.010)
s2 = max(s_strong, s_persistent)

d = [log(P_AC), sign(s2)*log(1+abs(s2)),
     log(P_AC)-log(1+max(s2,0)), P_AC/(1+max(s2,0))]
p_cons = sigmoid(Linear(standardize(d)))
p2 = 1 - (1-p_neural)*(1-p_cons)
```

The logarithm uses a positive numerical floor.

## Node3

Node3 estimates `ISRJ` conditional on the non-narrowband group. Its concepts
are the normalized pulse-compression envelope autocorrelation lag, temporal
energy concentration, and normalized complex IQ autocorrelation lag.

Its neural feature concatenates 512 STFT, 64 pulse-profile, 64 IQ-profile,
and 32 direct-concept features. A 640-to-128-to-3 auxiliary head provides
concept supervision.

```text
s_primary = (0.050 - c31) / 0.020
s_fallback = min((0.050 - c33) / 0.020, (c32 - 0.800) / 0.100)
l_physical = clip(max(s_primary,s_fallback) / 0.5, -12, 12)
l_neural = 6*tanh(l_neural_raw/6)
r_neural = 0.10 + 0.40*sigmoid(alpha)
p3 = sigmoid(l_physical + r_neural*l_neural)
```

## Routing

All three continuous node probabilities are used for final classification.
The output order is `NNJ, ISRJ, DFTJ, CJ`. The model selects the largest of
the four leaf probabilities. `hard_prediction` is a diagnostic hard-route
output; `prediction` is the soft-tree prediction used by the model.

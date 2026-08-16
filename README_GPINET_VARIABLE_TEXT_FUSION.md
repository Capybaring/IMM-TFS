# GPINet variable-to-report fusion

This patch replaces GPINet's previous 24-query text pooling with feature-level
fusion inside MTGNN.

## New data flow

```text
irregular numeric history
  -> GP mean/std on the numerical history grid
  -> Gauss-Hermite numerical features
  -> MTGNN temporal convolution

K timestamped radiology reports
  -> K precomputed BERT embeddings
  -> content projection + Time2Vec(original report timestamp)
  -> K report-event tokens (K stays K)

MTGNN variable hidden states
  -> each variable node queries the K report tokens
  -> gated variable-specific text message
  -> MTGNN graph propagation and subsequent blocks
  -> future-time decoder
```

The report set is never resampled to the 24-point GP grid. The text message
written into an MTGNN variable node is static patient context, not a claim that
a report was observed at every hour.

## Changed files

- `models/GPINet.py`: report-event encoder and in-backbone variable-text fusion.
- `main.py`: skips the unused generic `FusionModel` for native GPINet text and
  forwards `use_text_embeddings` explicitly.
- `lib/evaluation.py`: documents and keeps the native GPINet routing.
- `scripts/run_gpinet_fixed.sh`: exposes the internal attention-head and gate
  settings; removes misleading generic TTF/MMF arguments for GPINet.
- `tests/test_gpinet_variable_text_fusion.py`: shape, no-op, output-effect, and
  gradient checks.

## Run

Numeric-only:

```bash
./scripts/run_gpinet_fixed.sh -n 1000 --seed 1
```

Multimodal:

```bash
./scripts/run_gpinet_fixed.sh -n 1000 --text --seed 1
```

Optional native-text settings:

```bash
./scripts/run_gpinet_fixed.sh -n 1000 --text \
  --text-heads 1 \
  --text-gate-bias -1.0
```

`hid_dim` must be divisible by `text-heads`.

## Verification

```bash
python -m py_compile models/GPINet.py main.py lib/evaluation.py
bash -n scripts/run_gpinet_fixed.sh
python tests/test_gpinet_variable_text_fusion.py
```

The test verifies that:

1. Uni and Multi output shapes match;
2. real report events alter the output;
3. an all-empty report set is an exact no-op relative to Uni in evaluation;
4. gradients reach the text projection, variable-text attention, and MTGNN
   graph propagation modules.

Validation logs additionally report `text_gate_mean` and
`text_attention_entropy`. In this implementation a larger gate means a larger
text update; compare it together with correct/shuffled/zero-text experiments,
not as a standalone proof that the report semantics are useful.

Existing checkpoints trained with the old `GPTextPooler` are not compatible
with the new native text module and should not be resumed for Multi training.
Train the new Uni/Multi experiments from scratch.

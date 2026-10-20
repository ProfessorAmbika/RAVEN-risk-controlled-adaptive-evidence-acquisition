# RAVEN: risk-controlled adaptive evidence acquisition

Code for the paper *Knowing When Enough Has Been Seen: Risk-Controlled Adaptive Evidence Acquisition for
High-Resolution Visual Reasoning*.

RAVEN answers a question from a cheap view of the image first. A small router, run on that same pass, decides
whether the answer can stop there or more visual evidence (a higher-resolution view or a crop) should be acquired. The router's stopping threshold is
calibrated with Learn-then-Test (Hoeffding-Bentkus p-values, fixed-sequence testing). With probability at least
1 - delta over the calibration draw, the rate at which the early answer differs from the full-budget answer
(fidelity loss, no labels needed) stays at or below eps. An accuracy-gap loss, max(0, correct_full - correct_early), is also supported.

All results in the paper were produced on one PC with a single NVIDIA RTX 3090 (24 GB), batch size 1 unless
stated otherwise, bf16, SDPA attention.

## Installation

```
conda create -n raven python=3.11 && conda activate raven
pip install -r requirements.txt
```
`ENVIRONMENT.txt` lists the exact package versions, model revisions and dataset revision used for the paper
(written by `bash collect_env.sh`). Models and datasets download from Hugging Face on first use:

| What | Hugging Face id | Revision used |
|---|---|---|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` | `259a3b64` |
| LLaVA-OneVision-7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | `0d506805` |
| MME-RealWorld | `yifanzhang114/MME-RealWorld-lmms-eval` (dataset) | `f1573901` |
| V* Bench | `craigwu/vstar_bench` (dataset) | |
| HR-Bench | `DreamMr/HR-Bench` (dataset) | |

## Files

| File | Role |
|---|---|
| `ltt_calibrate.py` | The calibration step: Learn-then-Test with Hoeffding-Bentkus p-values over a threshold path |
| `analyze_gonogo.py` | Shared log utilities (`load`, `tensors`, `fit_logreg`, `evaluate`) and the multi-crop pilot analysis |
| `mmerw.py` | MME-RealWorld loader: stratified sample of 1,000 questions per domain, official prompt |
| `run_gonogo.py` | Qwen2.5-VL runs: single views (`--T 0`) or a view plus attention-ranked crops; logs features and hidden state |
| `run_backbone.py` | Single-view runs for InternVL3 and LLaVA-OneVision, and timed runs for any backbone |
| `run_vicrop.py` | ViCrop (relative attention), our reimplementation for Qwen2.5-VL |
| `run_zoomeye.py` | Zoom Eye baseline, run through the authors' code (clone github.com/om-ai-lab/ZoomEye) |
| `make_ladder.py`, `make_crop_ladder.py` | Join single-view logs into two-rung ladders with cumulative cost |
| `probe.py` | Main table: probe, confidence and random routers, each calibrated with LTT, over 100 splits |
| `audit_exact.py` | Exact audit of the guarantee: exact population risk, violation rate with Clopper-Pearson intervals |
| `baselines.py` | Plug-in threshold, conformal risk control, routers matched to RAVEN's cost, oracle |
| `latency_dist.py` | Per-question latency distribution of RAVEN against fixed views; routing and calibration overhead |
| `time_suite.py` | Repeated timing runs, crop-rung timing with attention maps, batched throughput |
| `compare_zoomeye.py` | Zoom Eye against fixed views, ViCrop and RAVEN on the same questions |
| `voi_test.py` | Appendix: a value-of-information head choosing between actions |
| `eps_sweep.py` | Sweep over eps used for the validity and frontier figures |
| `transfer.py`, `per_bench2.py`, `hrbench_groups.py` | Supplement: HR-Bench to V* transfer, per-benchmark pilot, HR-Bench duplicate groups |
| `collect_env.sh` | Writes `ENVIRONMENT.txt` |

Run all commands from this folder. Every run script appends one JSON line per question to `--out` and skips
questions that are already there, so an interrupted run resumes when the same command is run again.

## Settings used in the paper

- **Data.** MME-RealWorld, 1,000 questions per domain (5 domains, 5,000 questions), spread evenly over each domain's
  sub-tasks, seed 0. Greedy decoding, at most 4 new tokens, answer letter parsed by `run_gonogo.normalize`.
- **Views.**

  | Backbone | Rung 0 | Rung 1 |
  |---|---|---|
  | Qwen2.5-VL-7B | at most 1,024 visual tokens | at most 4,096 tokens, or ViCrop crop (1,024 tokens) plus the view |
  | InternVL3-8B | up to 4 tiles of 448 px plus thumbnail | up to 16 tiles |
  | LLaVA-OneVision-7B | long side resized to 384 px | long side 1,536 px |

  Cost is cumulative: an escalated question pays for both passes.
- **Router.** Hidden state of the last prompt token at the middle and last LLM layer of the rung-0 pass,
  standardised and reduced with PCA to 64 dimensions, plus 8 confidence features. Logistic regression with L2 0.1 and 3,000 steps.
- **Calibration.** 201 thresholds from 1.0 to 0.0, fixed-sequence testing, Hoeffding-Bentkus p-values, delta = 0.1,
  eps in {0.05, 0.10}, 300 calibration questions per domain.
- **Splits.** Per domain, 40% router training, 30% calibration, 30% test, 100 random splits (numpy seed 0).
  Exact audit: router seeds 1000-1002, 1,000-2,000 calibration draws.

## Reproducing the results

**1. Single-view logs** (MME-RealWorld, 1,000 questions per domain)
```
M=Qwen/Qwen2.5-VL-7B-Instruct
for T in 256 1024 4096; do
  python run_gonogo.py --model $M --bench mmerw --n 1000 --T 0 --global_tokens $T --save_hidden --out logs/mmerw_7b_global$T.jsonl
done
python run_vicrop.py --model $M --bench mmerw --n 1000 --global_tokens 1024 --crop_tokens 1024 --out logs/mmerw_7b_vicrop2.jsonl
python run_backbone.py --family internvl --tiles 4  --bench mmerw --n 1000 --out logs/mmerw_internvl8b_t4.jsonl
python run_backbone.py --family internvl --tiles 16 --bench mmerw --n 1000 --out logs/mmerw_internvl8b_t16.jsonl
python run_backbone.py --family llavaov --long_side 384  --bench mmerw --n 1000 --out logs/mmerw_llavaov7b_s384.jsonl
python run_backbone.py --family llavaov --long_side 1536 --bench mmerw --n 1000 --out logs/mmerw_llavaov7b_s1536.jsonl
```

**2. Ladders**
```
python make_ladder.py logs/mmerw_7b_global1024.jsonl logs/mmerw_7b_global4096.jsonl --out logs/mmerw_7b_l1024.jsonl
python make_ladder.py logs/mmerw_7b_global256.jsonl  logs/mmerw_7b_global4096.jsonl --out logs/mmerw_7b_l256.jsonl
python make_crop_ladder.py logs/mmerw_7b_l1024.jsonl logs/mmerw_7b_vicrop2.jsonl --out logs/mmerw_7b_lvicrop_reuse.jsonl
python make_ladder.py logs/mmerw_internvl8b_t4.jsonl logs/mmerw_internvl8b_t16.jsonl --out logs/mmerw_internvl8b_ladder.jsonl
python make_ladder.py logs/mmerw_llavaov7b_s384.jsonl logs/mmerw_llavaov7b_s1536.jsonl --out logs/mmerw_llavaov7b_ladder.jsonl
```

**3. Main results, guarantee audit, baselines** (CPU only; use `OMP_NUM_THREADS=1` for `baselines.py`)
```
mkdir -p results
for L in 7b_l1024 7b_l256 7b_lvicrop_reuse internvl8b_ladder llavaov7b_ladder; do
  python probe.py --train logs/mmerw_$L.jsonl --by_domain > results/probe_$L.txt
done
python audit_exact.py --ladder l1024=logs/mmerw_7b_l1024.jsonl --ladder l256=logs/mmerw_7b_l256.jsonl \
    --ladder crop=logs/mmerw_7b_lvicrop_reuse.jsonl --B 2000 --seeds 3 --out results/audit_exact.json
python baselines.py --ladder logs/mmerw_7b_l1024.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl \
    --out results/baselines_qwen7b_l1024.json
python eps_sweep.py --logs logs --out results/sweep.json
python voi_test.py --l1024 logs/mmerw_7b_l1024.jsonl --vicrop logs/mmerw_7b_vicrop2.jsonl --out results/voi_test.json
```
`baselines.py` and `audit_exact.py` take the other ladders in the same way.

**4. Measured time** (GPU; 100 questions per domain)
```
python run_backbone.py --family qwen --tokens 1024 --bench mmerw --n 100 --out logs/time_qwen_1024.jsonl
python run_backbone.py --family qwen --tokens 4096 --bench mmerw --n 100 --out logs/time_qwen_4096.jsonl
python latency_dist.py --ladder logs/mmerw_7b_l1024.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl \
    --out results/latency_dist_qwen7b.json
python time_suite.py --parts repeat crop batch --out logs/timesuite_qwen
```

**5. Zoom Eye**
```
git clone https://github.com/om-ai-lab/ZoomEye.git && python -m spacy download en_core_web_sm
python run_zoomeye.py --zoomeye ZoomEye --per_domain 100 --out logs/mmerw_zoomeye_q7b.jsonl
python compare_zoomeye.py --zoomeye logs/mmerw_zoomeye_q7b.jsonl --ladder logs/mmerw_7b_l1024.jsonl \
    --crop logs/mmerw_7b_lvicrop_reuse.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl \
    --out results/zoomeye_vs_raven_qwen7b.json
```
Each image Zoom Eye builds is capped at 4,096 visual tokens (`--tokens`), the budget of the fixed high view. This
is lower than Zoom Eye's own default. FlashAttention-2 is replaced by SDPA.

**6. Pilot and supplement** (V*, HR-Bench)
```
python run_gonogo.py --bench vstar     --model $M --out logs/vstar_qwen7b.jsonl
python run_gonogo.py --bench hrbench4k --model $M --out logs/hrbench4k_qwen7b.jsonl
python run_gonogo.py --bench hrbench8k --model $M --out logs/hrbench8k_qwen7b.jsonl
python hrbench_groups.py --out logs/hrbench_groups.json
python transfer.py --train logs/hrbench4k_qwen7b.jsonl logs/hrbench8k_qwen7b.jsonl --eval logs/vstar_qwen7b.jsonl
python per_bench2.py logs/vstar_qwen7b.jsonl logs/hrbench4k_qwen7b.jsonl logs/hrbench8k_qwen7b.jsonl
```

## Using the calibration on your own logs

```python
import numpy as np
from ltt_calibrate import calibrate
# score[i, t]: router score after step t (higher = stop); loss[i, t] in [0, 1]; cost[i, t]: cumulative cost.
# The last step is the full budget: its score is 1 and its loss is 0.
cal = calibrate(score, loss, cost, eps=0.05, delta=0.1)
stop_now = new_score >= cal.lam
```

## Citation

To be added on publication.

## License

See `LICENSE` (to be added before release). Zoom Eye, ViCrop and the models and datasets above keep their own
licenses.

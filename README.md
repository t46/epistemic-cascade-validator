# Epistemic Cascade Validator

A confidence scoring system that detects and prevents epistemic cascade contamination in autonomous research pipelines.

## The Problem

In autoresearch pipelines (e.g., running ~100 ML experiments per night on an H100), a "keep" result that doesn't reproduce is **worse than a discard**. When a false positive propagates downstream through a chain of experiments, it creates **cascade contamination**: each subsequent experiment builds on a false premise, compounding wasted compute and generating misleading results.

This is the reproducibility crisis in miniature. In traditional science, a retracted paper that continues to be cited creates cascading harm. In autoresearch, the same dynamic plays out at machine speed -- a false positive at step 3 of a 15-step experiment chain can corrupt all downstream work within hours.

## The Solution

The Epistemic Cascade Validator (ECV) assigns **Bayesian confidence scores** to experimental results, enabling downstream experiments to treat prior outputs as **uncertain premises** rather than ground truth.

### How it works

1. **Confidence Scoring**: Each experimental result receives a 0.0-1.0 confidence score computed from four signals:
   - **Reproduction rate** (40% weight): Beta-Binomial posterior from replication trials
   - **Statistical evidence** (25%): Transformed p-value with sample size adjustment
   - **Effect size** (20%): Calibrated sigmoid mapping of Cohen's d
   - **Code quality** (15%): Lint, test coverage, complexity metrics

2. **Cascade Gating**: A decision engine evaluates whether each upstream result should be used as a premise. Results below the confidence threshold are **gated** -- downstream experiments treat them as if the upstream reported negative, breaking the contamination chain.

3. **Uncertainty Compounding**: In a chain A->B->C, the effective confidence of C's premise is the product of upstream confidences. A chain of 0.7 x 0.8 = 0.56 means the final result inherits compounded uncertainty.

### Design Philosophy

This system embodies the principle from research social infrastructure design: **quality-conditional settlement**. Just as a financial escrow holds funds until verification conditions are met, the ECV holds experimental results in an "uncertain" state until reproduction evidence accumulates. The confidence score is a continuously-updated escrow balance.

## Results

Running 100 seeds across two cascade scenarios:

| Metric | Linear Chain (L=15) | Branching Tree (D=4, B=2) |
|--------|--------------------|-----------------------------|
| Contamination reduction | **~78%** | **~79%** |
| FDR reduction | **~69%** | **~34%** |
| Mean gated nodes | 1.4 | 3.6 |
| Statistical significance | p < 1e-17 | p < 1e-31 |

The sensitivity analysis shows a sharp transition: at threshold 0.4, contamination drops by 72%; at 0.5, by 98%; at 0.6+, contamination is virtually eliminated.

## Architecture

```
ecv/
  __init__.py          # Package exports
  confidence.py        # Bayesian confidence scoring model
  cascade.py           # Cascade simulator (DAG-based)
  decision.py          # Confidence-aware decision engine
  comparison.py        # Statistical comparison framework
  visualization.py     # Matplotlib visualization
  adapters/            # Real-data source adapters
    __init__.py
    base.py            # BaseAdapter interface & AdaptedResult
    evaluator.py       # auto-research-evaluator (14 experiments)
    ara.py             # ARA belief store (claims + ledger)
    vanilla.py         # vanilla autoresearch hypothesis pipeline
main.py                # Simulation demo runner with CLI args
validate_real_data.py  # Real-data validation runner
output/                # Generated figures and results
```

## Usage

```bash
# Install dependencies
uv sync

# Run the full demo (generates all figures and statistics)
uv run main.py

# Customize parameters
uv run main.py --seeds 200 --chain-length 20 --fpr 0.20 --threshold 0.5
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | 100 | Number of random seeds for statistical comparison |
| `--chain-length` | 15 | Length of the linear experiment chain |
| `--fpr` | 0.15 | False positive rate per experiment |
| `--threshold` | 0.4 | Confidence threshold for gating |

### Programmatic API

```python
from ecv import ConfidenceScorer, CascadeSimulator, DecisionEngine
from ecv.confidence import EvidencePacket

# Score a single result
scorer = ConfidenceScorer()
evidence = EvidencePacket(
    reproduction_successes=4,
    reproduction_attempts=5,
    effect_size=0.6,
    p_value=0.003,
    sample_size=100,
    code_has_tests=True,
    code_test_coverage=0.75,
)
score = scorer.score(evidence)
print(score)  # ConfidenceScore(overall=0.820, uncertainty=0.160)

# Decide whether to use as premise
engine = DecisionEngine(risk_tolerance="medium")
decision = engine.should_use_as_premise(score, threshold=0.4)
print(decision)  # Decision(use=True, risk=medium, ...)

# Run a cascade simulation
sim = CascadeSimulator(seed=42)
sim.build_linear_chain(length=10, false_positive_rate=0.15, contamination_start=3)
result = sim.run_with_scoring(confidence_threshold=0.4)
print(f"Contamination rate: {result.contamination_rate:.3f}")
```

## Real-Data Validation

In addition to the simulation, ECV has been validated against three real autoresearch pipeline outputs. Each data source required a custom adapter to map its output format to ECV's `EvidencePacket` schema.

### Data Sources

| Source | Type | N | Description |
|--------|------|---|-------------|
| auto-research-evaluator | Paper meta-evaluation | 13 | Multi-phase analysis of autoresearch-generated papers (exp-2025-11-19 to exp-2025-12-02). Scores for methodology, statistical validity, reproducibility, etc. |
| ARA belief store | Literature-review claims | 4 | Claims with confidence levels, evidence (for/against), fidelity ratings, and explicit dependency chains |
| vanilla autoresearch | Hypothesis pipeline | 6 | Hypotheses with status (proposed/rejected), self-assessed confidence, and estimated novelty |

### Results

```
Cross-Source Comparison:
  Source                            N    Mean     Med     Std  Gated%
  auto-research-evaluator          13  0.6254  0.6250  0.0460    0.0%
  autonomous-research-agent         4  0.5082  0.5062  0.0068    0.0%
  vanilla-autoresearch              6  0.5092  0.4823  0.2046   50.0%
```

Key observations:
- **Evaluator experiments** cluster in the medium-confidence range (0.53-0.70), reflecting that all are published papers with reasonable quality
- **ARA claims** are tightly clustered around 0.50, reflecting F1-fidelity evidence (abstract-only reads) with moderate self-reported confidence
- **Vanilla hypotheses** show the widest spread (0.30-0.75): rejected hypotheses correctly receive low scores (~0.31), while proposed hypotheses with higher confidence and novelty receive higher scores (~0.73). The 50% gating rate correctly filters rejected hypotheses
- **Cascade analysis** on ARA's dependency chain shows compounded confidence dropping from 0.51 (root claims) to 0.13 (downstream claim-004 which depends on three upstream claims), demonstrating uncertainty propagation in real epistemic chains

### Usage

```bash
# Run validation against all data sources
uv run validate_real_data.py

# Verbose output with per-result details
uv run validate_real_data.py --verbose

# Single source
uv run validate_real_data.py --source evaluator
uv run validate_real_data.py --source ara
uv run validate_real_data.py --source vanilla

# Custom threshold
uv run validate_real_data.py --threshold 0.5
```

### Adapter Design

Each adapter implements `BaseAdapter` with a `load(path) -> list[AdaptedResult]` method. The `AdaptedResult` preserves provenance (source file, raw scores, mapping notes) alongside the derived `EvidencePacket`, enabling audit of the proxy mappings.

```python
from ecv.adapters import AutoResearchEvaluatorAdapter
from ecv.confidence import ConfidenceScorer

adapter = AutoResearchEvaluatorAdapter()
results = adapter.load(Path("~/unktok/dev/auto-research-evaluator"))

scorer = ConfidenceScorer()
for r in results:
    score = scorer.score(r.evidence)
    print(f"{r.experiment_id}: {score.overall:.3f} ({r.mapping_notes[0]})")
```

### Limitations

1. **Proxy mapping, not ground truth**: All EvidencePacket fields are derived from available metadata, not actual reproduction trials or effect size measurements. The scores reflect "how well-documented and self-consistent" the research is, not "how likely to reproduce."

2. **Small sample (N=23)**: Insufficient for statistical calibration conclusions. The validation shows the system runs on real data and produces differentiated scores, but cannot assess accuracy.

3. **No ground truth for gating decisions**: We cannot measure precision/recall of the gating decisions because we don't know which results are true/false positives.

4. **Evaluator circularity**: The evaluator scores are LLM-generated assessments. Using LLM-assessed quality to validate an LLM-quality-scoring system has inherent circularity.

5. **Shallow cascade depth**: The ARA chain has 2-3 levels of dependency. We cannot validate cascade behavior at the 10-15 node depths used in simulation.

## Connection to the Reproducibility Crisis

This prototype addresses the micro-scale version of a macro-scale problem. The traditional reproducibility crisis unfolds over years: a paper with a false positive is published, cited, built upon, and eventually (maybe) retracted -- but its downstream effects persist. In autoresearch, the same dynamic unfolds in hours or days.

The ECV approach maps to several principles from research infrastructure design:

- **Truth-incentive compatibility**: Confidence scoring makes unreproducible results automatically penalized
- **Replication incentives**: The cost of downstream failure from unreproduced results creates natural incentive to verify
- **Failure tolerance**: Low-confidence results are treated as "needs more evidence" rather than "failed", preserving information value

## License

MIT

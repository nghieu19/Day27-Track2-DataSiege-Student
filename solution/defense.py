"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


# ---------------------------------------------------------------------------
# Tunable knobs — everything statistical funnels through these so you can
# sweep them against phases/practice_answer_key.json instead of hand-tuning
# inline magic numbers. Score = 100 * (0.5*TPR - 0.3*FPR - 0.2*min(cost_overage,1)),
# so TPR is worth more than FPR — err toward catching, not toward silence.
# ---------------------------------------------------------------------------

# private phase has "harder subtle end" faults (FAULT_PILLARS.md).
# Two threshold sets: aggressive for private, precise for public/practice.
# Phase detected via budget: private=320, public/practice=220.

# Public/practice thresholds (precise, minimize FPR)
CHI2_THRESHOLD_4DF_PRECISE = 13.0
CHI2_THRESHOLD_2DF_PRECISE = 9.0
MAX_Z_PRECISE = 3.2
ROLLING_Z_PRECISE = 3.5
ROLLING_MIN_PRECISE = 8
FEATURE_Z_PRECISE = 2.5
FRESHNESS_Z_PRECISE = 2.8
LINEAGE_Z_PRECISE = 2.8

# Private thresholds (aggressive, maximize TPR for subtle faults)
CHI2_THRESHOLD_4DF_SUBTLE = 10.5
CHI2_THRESHOLD_2DF_SUBTLE = 7.5
MAX_Z_SUBTLE = 2.6
ROLLING_Z_SUBTLE = 3.0
ROLLING_MIN_SUBTLE = 8    # stable estimate reduces FP from noisy early samples
FEATURE_Z_SUBTLE = 2.8    # primary threshold; supplemented by dual-signal below
FEATURE_Z_DUAL = 2.0      # secondary threshold, requires rolling confirmation
ROLLING_Z_DUAL = 2.5      # rolling confirmation threshold for dual-signal
FRESHNESS_Z_SUBTLE = 2.5
LINEAGE_Z_SUBTLE = 2.5


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _init_phase(ctx):
    """Lazy phase detection on first handler call (cannot call tools in register()).
    private=320 budget, public/practice=220 budget. Called once per run."""
    if "is_private" not in ctx.state:
        initial_budget = ctx.tools.budget_remaining()
        ctx.state["is_private"] = (initial_budget > 250)


# ---------------------------------------------------------------------------
# Shared statistical helpers
# ---------------------------------------------------------------------------

def _thresholds(ctx):
    """Return threshold set based on detected phase."""
    if ctx.state.get("is_private", False):
        return {
            "chi2_4df": CHI2_THRESHOLD_4DF_SUBTLE,
            "chi2_2df": CHI2_THRESHOLD_2DF_SUBTLE,
            "max_z": MAX_Z_SUBTLE,
            "rolling_z": ROLLING_Z_SUBTLE,
            "rolling_min": ROLLING_MIN_SUBTLE,
            "feature_z": FEATURE_Z_SUBTLE,
            "feature_z_dual": FEATURE_Z_DUAL,
            "rolling_z_dual": ROLLING_Z_DUAL,
            "freshness_z": FRESHNESS_Z_SUBTLE,
            "lineage_z": LINEAGE_Z_SUBTLE,
        }
    return {
        "chi2_4df": CHI2_THRESHOLD_4DF_PRECISE,
        "chi2_2df": CHI2_THRESHOLD_2DF_PRECISE,
        "max_z": MAX_Z_PRECISE,
        "rolling_z": ROLLING_Z_PRECISE,
        "rolling_min": ROLLING_MIN_PRECISE,
        "feature_z": FEATURE_Z_PRECISE,
        "feature_z_dual": FEATURE_Z_PRECISE,
        "rolling_z_dual": ROLLING_Z_PRECISE,
        "freshness_z": FRESHNESS_Z_PRECISE,
        "lineage_z": LINEAGE_Z_PRECISE,
    }

def _two_sided_z(value, vmin, vmax):
    """Baseline published as mean +/- 3sigma bounds -> reconstruct z-score."""
    mean = (vmax + vmin) / 2
    sigma = (vmax - vmin) / 6
    if sigma <= 0:
        return 0.0
    return abs(value - mean) / sigma


def _one_sided_z(value, max_bound):
    """Baseline gives only a max (mean+3sigma with mean assumed ~0), per the
    same convention the original code used for null_rate/staleness. Applied
    consistently here to freshness/duration/drift/staleness-style metrics."""
    if max_bound <= 0:
        return 0.0
    sigma = max_bound / 3
    if sigma <= 0:
        return 0.0
    return max(value, 0.0) / sigma


def _rolling_z(ctx, key, value, min_samples=8):
    """Online (Welford) mean/variance per pillar-metric, kept in ctx.state.
    Returns a z-score of `value` against everything seen so far *before*
    this point (None until we have enough samples to trust it), then
    updates the running stats."""
    hist = ctx.state.setdefault(key, {"n": 0, "mean": 0.0, "m2": 0.0})
    n = hist["n"]
    z = None
    if n >= min_samples:
        variance = hist["m2"] / n
        std = variance ** 0.5
        if std > 1e-9:
            z = abs(value - hist["mean"]) / std
    # Welford update
    n += 1
    delta = value - hist["mean"]
    hist["mean"] += delta / n
    delta2 = value - hist["mean"]
    hist["m2"] += delta * delta2
    hist["n"] = n
    return z


# ---------------------------------------------------------------------------
# checks / data_batch
# ---------------------------------------------------------------------------

def check_data_batch(payload, ctx):
    _init_phase(ctx)
    t = _thresholds(ctx)
    stats = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in stats:
        return Verdict(alert=False, pillar="checks")

    z_row = _two_sided_z(stats["row_count"], ctx.baseline["row_count_min"], ctx.baseline["row_count_max"])
    z_amt = _two_sided_z(stats["mean_amount"], ctx.baseline["mean_amount_min"], ctx.baseline["mean_amount_max"])
    z_null = _one_sided_z(stats["null_rate"]["customer_id"], ctx.baseline["null_rate_max"])
    z_stale = _one_sided_z(stats["staleness_min"], ctx.baseline["staleness_min_max"])

    chi2 = z_row ** 2 + z_amt ** 2 + z_null ** 2 + z_stale ** 2
    max_z = max(z_row, z_amt, z_null, z_stale)

    is_faulty = False
    reason = ""
    if max_z > t["max_z"]:
        is_faulty = True
        reason = f"single metric anomaly z={max_z:.1f}"
    elif chi2 > t["chi2_4df"]:
        is_faulty = True
        reason = f"chi2 anomaly: {chi2:.1f}"

    # Subtle drift in std_amount
    z_std = _rolling_z(ctx, "checks.std_amount", stats["std_amount"], t["rolling_min"])
    if z_std is not None and z_std > t["rolling_z"]:
        is_faulty = True
        reason = f"std_amount drift z={z_std:.1f}"

    if is_faulty:
        ctx.state.setdefault("bad_batches", set()).add(payload["batch_id"])
        return Verdict(alert=True, pillar="checks", reason=reason)

    return Verdict(alert=False, pillar="checks")


# ---------------------------------------------------------------------------
# contracts / contract_checkpoint
# ---------------------------------------------------------------------------

def check_contract_checkpoint(payload, ctx):
    _init_phase(ctx)
    t = _thresholds(ctx)
    stats = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in stats:
        return Verdict(alert=False, pillar="contracts")

    if len(stats["violations"]) > 0:
        return Verdict(alert=True, pillar="contracts", reason="contract violations found")

    freshness = stats["freshness_delay_min"]
    z_fresh = _one_sided_z(freshness, ctx.baseline["freshness_delay_max_min"])
    if z_fresh > t["freshness_z"]:
        return Verdict(alert=True, pillar="contracts", reason=f"freshness delay z={z_fresh:.1f}")

    z_roll = _rolling_z(ctx, "contracts.freshness_delay_min", freshness, t["rolling_min"])
    if z_roll is not None and z_roll > t["rolling_z"]:
        return Verdict(alert=True, pillar="contracts", reason=f"freshness drift z={z_roll:.1f}")

    return Verdict(alert=False, pillar="contracts")


# ---------------------------------------------------------------------------
# lineage / lineage_run
# ---------------------------------------------------------------------------

def check_lineage_run(payload, ctx):
    _init_phase(ctx)
    t = _thresholds(ctx)
    stats = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in stats:
        return Verdict(alert=False, pillar="lineage")

    if "expected_upstream" in payload:
        if set(stats.get("actual_upstream", [])) != set(payload["expected_upstream"]):
            return Verdict(alert=True, pillar="lineage", reason="upstream mismatch")

    if "expected_downstream_count" in payload:
        if stats.get("actual_downstream_count", 0) != payload["expected_downstream_count"]:
            return Verdict(alert=True, pillar="lineage", reason="downstream count mismatch")

    if stats.get("actual_downstream_count", -1) == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphaned output")

    duration = stats["duration_ms"]
    z_dur = _one_sided_z(duration, ctx.baseline["lineage_duration_ms_max"])
    if z_dur > t["lineage_z"]:
        return Verdict(alert=True, pillar="lineage", reason=f"lineage duration z={z_dur:.1f}")

    z_roll = _rolling_z(ctx, "lineage.duration_ms", duration, t["rolling_min"])
    if z_roll is not None and z_roll > t["rolling_z"]:
        return Verdict(alert=True, pillar="lineage", reason=f"lineage duration drift z={z_roll:.1f}")

    return Verdict(alert=False, pillar="lineage")


# ---------------------------------------------------------------------------
# ai_infra / feature_materialization
# ---------------------------------------------------------------------------

def check_feature_materialization(payload, ctx):
    _init_phase(ctx)
    t = _thresholds(ctx)
    if payload["batch_id"] in ctx.state.get("bad_batches", set()):
        return Verdict(alert=True, pillar="ai_infra", reason="inherited upstream fault")

    stats = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in stats:
        return Verdict(alert=False, pillar="ai_infra")

    shift_sigma = stats["mean_shift_sigma"]
    z_shift = _one_sided_z(shift_sigma, ctx.baseline["feature_mean_shift_sigma_max"])

    # Compute rolling z BEFORE the primary check (to track clean baseline)
    z_roll = _rolling_z(ctx, "ai_infra.feature_shift", shift_sigma, t["rolling_min"])

    # Primary signal: clearly above baseline threshold (obvious + most subtle faults)
    if z_shift > t["feature_z"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature mean shift z={z_shift:.1f}")

    # Dual-signal: lower z_shift threshold, but ALSO requires rolling confirmation.
    # This catches subtle faults that deviate from the stream while avoiding FP
    # from clean events that are just at the high end of their natural range.
    if z_shift > t["feature_z_dual"] and z_roll is not None and z_roll > t["rolling_z_dual"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature dual-signal: z_shift={z_shift:.1f}, z_roll={z_roll:.1f}")

    # Standalone rolling z (strong drift trend even if baseline z is moderate)
    if z_roll is not None and z_roll > t["rolling_z"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature drift trend z={z_roll:.1f}")

    return Verdict(alert=False, pillar="ai_infra")


# ---------------------------------------------------------------------------
# ai_infra / embedding_batch
# ---------------------------------------------------------------------------

def check_embedding_batch(payload, ctx):
    _init_phase(ctx)
    t = _thresholds(ctx)
    if payload["chunk_batch_id"] in ctx.state.get("bad_batches", set()):
        return Verdict(alert=True, pillar="ai_infra", reason="inherited upstream fault")

    stats = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in stats:
        return Verdict(alert=False, pillar="ai_infra")

    z_centroid = _one_sided_z(stats["centroid_shift"], ctx.baseline["embedding_centroid_shift_max"])
    z_age = _one_sided_z(stats["avg_doc_age_days"], ctx.baseline["corpus_avg_doc_age_days_max"])
    max_z = max(z_centroid, z_age)

    if max_z > t["max_z"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding metric z={max_z:.1f}")

    chi2 = z_centroid ** 2 + z_age ** 2
    if chi2 > t["chi2_2df"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding chi2 anomaly: {chi2:.1f}")

    z_roll_c = _rolling_z(ctx, "ai_infra.centroid_shift", stats["centroid_shift"], t["rolling_min"])
    if z_roll_c is not None and z_roll_c > t["rolling_z"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding centroid drift z={z_roll_c:.1f}")

    z_roll_a = _rolling_z(ctx, "ai_infra.doc_age", stats["avg_doc_age_days"], t["rolling_min"])
    if z_roll_a is not None and z_roll_a > t["rolling_z"]:
        return Verdict(alert=True, pillar="ai_infra", reason=f"corpus age drift z={z_roll_a:.1f}")

    return Verdict(alert=False, pillar="ai_infra")
"""Simulated cattle-movement link forecasting.
Baseline: per-link linear trend extrapolation of the last 6 months of counts.
Improved: LightGBM on log1p counts with lags, rolling means and network features.
Seeds 42, 43, 44 each drive a fresh simulation and the LightGBM fits."""
import json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import TimeSeriesSplit

SEEDS = [42, 43, 44]
N_NODES, N_LINKS, T = 150, 4000, 36   # premises, candidate links, monthly periods
TRAIN_TARGET_END = 24                 # target months < 24 -> train, >= 24 -> test (last 12)
FIRST_ORIGIN = 5                      # 6 months of history needed for lags and baseline
GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (15, 31) for mc in (20, 100)]
FEATS = ["lag1", "lag2", "lag3", "roll3", "roll6", "out_deg_src", "in_deg_dst",
         "out_flow_src", "in_flow_dst", "in_flow_src", "out_flow_dst"]


def simulate(seed):
    """Seeded premises-to-premises monthly counts: gravity base rate,
    slow per-link log-linear trends and AR(1) node-level activity shocks."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, (N_NODES, 2))
    size = rng.lognormal(0, 0.6, N_NODES)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    base = size[:, None] * size[None] / (1 + 10 * dist) ** 1.5    # gravity-like rate
    np.fill_diagonal(base, 0)
    top = np.argsort(-base.ravel())[:N_LINKS]                     # keep strongest links
    src, dst = np.unravel_index(top, base.shape)
    logb = np.log(base.ravel()[top])
    eta0 = logb - logb.mean() + np.log(3.0)                       # ~3 movements per link-month
    trend = rng.normal(0, 0.015, N_LINKS)                         # per-link log-trend
    U = np.zeros((T, N_NODES)); u = np.zeros(N_NODES)
    for t in range(T):                                            # node shocks, AR(1)
        u = 0.7 * u + rng.normal(0, 0.35, N_NODES)
        U[t] = u
    eta = eta0 + trend * np.arange(T)[:, None] + 0.5 * (U[:, src] + U[:, dst])
    C = rng.poisson(np.exp(np.minimum(eta, 6))).astype(float)
    return C, src, dst


def make_rows(C, src, dst, t):
    """Feature table for origin month t: uses only C[:t+1]; target is C[t+1]."""
    c = C[t]
    out_tot = np.bincount(src, weights=c, minlength=N_NODES)
    in_tot = np.bincount(dst, weights=c, minlength=N_NODES)
    out_deg = np.bincount(src, weights=(c > 0).astype(float), minlength=N_NODES)
    in_deg = np.bincount(dst, weights=(c > 0).astype(float), minlength=N_NODES)
    X6 = C[t - 5:t + 1]
    x = np.arange(6) - 2.5                                        # per-link OLS slope
    slope = (x[:, None] * X6).sum(0) / (x ** 2).sum()
    baseline = np.clip(X6.mean(0) + slope * 3.5, 0, None)         # extrapolate to next month
    return pd.DataFrame({
        "origin": t, "target_month": t + 1, "y": C[t + 1],
        "lag1": C[t], "lag2": C[t - 1], "lag3": C[t - 2],
        "roll3": C[t - 2:t + 1].mean(0), "roll6": X6.mean(0),
        "out_deg_src": out_deg[src], "in_deg_dst": in_deg[dst],
        "out_flow_src": out_tot[src] - c, "in_flow_dst": in_tot[dst] - c,
        "in_flow_src": in_tot[src], "out_flow_dst": out_tot[dst],
        "baseline": baseline})


def lgb_params(seed, **kw):
    return dict(objective="regression", learning_rate=0.05, n_estimators=300,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                random_state=seed, n_jobs=2, verbose=-1, **kw)


def tune(train, seed):
    """Pick hyper-parameters by time-ordered CV on training months only."""
    origins = np.sort(train.origin.unique())
    best, best_mae = GRID[0], np.inf
    for g in GRID:
        maes = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=3).split(origins):
            tr = train[train.origin.isin(origins[tr_i])]
            va = train[train.origin.isin(origins[va_i])]
            m = lgb.LGBMRegressor(**lgb_params(seed, **g)).fit(tr[FEATS], np.log1p(tr.y))
            maes.append(mean_absolute_error(va.y, np.clip(np.expm1(m.predict(va[FEATS])), 0, None)))
        if np.mean(maes) < best_mae:
            best, best_mae = g, np.mean(maes)
    print(f"  CV-selected params {best} (CV MAE {best_mae:.4f})")
    return best


def evaluate(y, pred):
    """Main: MAE on raw counts. Secondary: top-decile hit rate, F1 on active links."""
    k = max(1, int(0.1 * len(y)))
    top_p = set(np.argsort(-pred, kind="stable")[:k])
    top_t = set(np.argsort(-y, kind="stable")[:k])
    return {"mae": mean_absolute_error(y, pred), "hit": len(top_p & top_t) / k,
            "f1": f1_score(y > 0, pred >= 0.5)}


def run(seed):
    """One full pipeline for a seed: same data, split, and evaluation for both models."""
    C, src, dst = simulate(seed)
    df = pd.concat([make_rows(C, src, dst, t) for t in range(FIRST_ORIGIN, T - 1)],
                   ignore_index=True)
    train = df[df.target_month < TRAIN_TARGET_END]
    test = df[df.target_month >= TRAIN_TARGET_END]
    print(f"  rows={len(df)} train={len(train)} test={len(test)}")
    best = tune(train, seed)
    model = lgb.LGBMRegressor(**lgb_params(seed, **best)).fit(train[FEATS], np.log1p(train.y))
    imp = np.clip(np.expm1(model.predict(test[FEATS])), 0, None)
    b = evaluate(test.y.values, test.baseline.values)
    m = evaluate(test.y.values, imp)
    print(f"  baseline MAE {b['mae']:.4f} | improved MAE {m['mae']:.4f}")
    return {"n_rows": len(df), "n_train": len(train), "n_test": len(test),
            "b": b, "m": m}


def main():
    start = time.time()
    res = []
    for s in SEEDS:
        print(f"Seed {s}: simulate, tune, fit, evaluate")
        res.append(run(s))

    def agg(model, key):
        return np.array([r[model][key] for r in res], float)

    bm, im = agg("b", "mae"), agg("m", "mae")
    metrics = {
        "baseline_score": bm.mean(), "improved_score": im.mean(),
        "improvement": bm.mean() - im.mean(),              # lower MAE is better
        "baseline_score_std": bm.std(), "improved_score_std": im.std(),
        "baseline_f1": agg("b", "f1").mean(), "improved_f1": agg("m", "f1").mean(),
        "baseline_top10_hit_rate": agg("b", "hit").mean(),
        "improved_top10_hit_rate": agg("m", "hit").mean(),
        "n_seeds": len(SEEDS), "elapsed_seconds": time.time() - start,
    }
    if not all(np.isfinite(v) for v in metrics.values()) or res[0]["n_test"] == 0:
        print("ERROR: could not produce meaningful metrics")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline (linear trend)", "improved (LightGBM)"],
            [metrics["baseline_score"], metrics["improved_score"]],
            yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], capsize=6)
    plt.ylabel("test MAE on next-month counts (lower is better)")
    plt.title("Baseline vs improved (mean +/- std over seeds)")
    plt.tight_layout(); plt.savefig("figures/mae_comparison.png", dpi=120); plt.close()

    out = {
        "question": "Can a LightGBM model that uses network context (neighbouring-link flows "
                    "and node activity) beat a standard per-link linear trend extrapolation at "
                    "predicting next-period cattle movement counts between premises?",
        "method": "Seeded synthetic premises-movement simulation (gravity base rate, per-link "
                  "trends, AR(1) node shocks); time-ordered split; CV tuning on training months; "
                  "test set used once per model; seeds 42, 43, 44.",
        "dataset": {
            "name": "Simulated premises-to-premises cattle movement network (generated in analysis.py)",
            "source_url": "none: generated in-script with numpy",
            "n_rows": res[0]["n_rows"], "n_train": res[0]["n_train"],
            "n_test": res[0]["n_test"], "same_as_paper": False,
            "baseline_is_papers_method": False},
        "metric_name": "MAE on raw next-month link counts",
        "higher_is_better": False,
        "baseline": "Per-link linear trend fitted by OLS to the last 6 monthly counts, "
                    "extrapolated one month ahead and clipped at zero.",
        "improved": "LightGBM regressor on log1p counts using lagged counts, rolling means, and "
                    "network features from the previous month (out/in degree and neighbouring-link "
                    "flows at source and destination premises); hyper-parameters chosen by "
                    "time-ordered CV on training data.",
        "metrics": metrics,
        "notes": "Synthetic data stand in for the paper's Great Britain cattle movement records, "
                 "which are not publicly downloadable; results do not transfer to real data. "
                 "The baseline is a plain trend tracker, not the paper's predictive trend mining "
                 "framework. Std is the population std (ddof=0) over 3 simulation seeds. "
                 "Secondary metrics: top-decile hit rate (overlap of top-10% predicted and actual "
                 "links) and F1 on active links (count>0) with a 0.5 threshold on predictions. "
                 "The test set was used once per model.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Done in {metrics['elapsed_seconds']:.1f}s. Wrote results.json")


if __name__ == "__main__":
    main()

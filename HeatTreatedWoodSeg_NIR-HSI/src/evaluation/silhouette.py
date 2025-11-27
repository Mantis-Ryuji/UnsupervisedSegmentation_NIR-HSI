from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import colorcet as cc

# =========================================================
# Silhouette 可視化
# =========================================================
def plot_silhouette_samples(
    silhouette_samples: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    *,
    max_bars_per_cluster: int = 5000,
    subsample_method: str = "stride",
    seed: Optional[int] = 42,
) -> None:
    """
    クラスタ別のシルエット係数を帯状に可視化して保存/表示する。

    Parameters
    ----------
    silhouette_samples : (N,) array
        各サンプルのシルエット係数。
    labels : (N,) array
        クラスタラベル。
    save_path : str or Path, optional
        保存先パス。None の場合は保存しない。
    show : bool, default False
        True の場合は plt.show() を呼ぶ。
    max_bars_per_cluster : int, default 5000
        1 クラスタあたりの最大棒本数（サブサンプリング上限）。
    subsample_method : {"stride", "random"}, default "stride"
        サブサンプリング方法。
    seed : int or None, default 42
        subsample_method="random" のときの乱数シード。
    """
    silhouette_samples = np.asarray(silhouette_samples)
    labels = np.asarray(labels)
    if silhouette_samples.ndim != 1 or labels.ndim != 1:
        raise ValueError("silhouette_samples と labels は 1 次元配列である必要があります。")
    if silhouette_samples.size != labels.size:
        raise ValueError("silhouette_samples と labels の要素数が一致しません。")

    if subsample_method not in {"stride", "random"}:
        raise ValueError("subsample_method must be 'stride' or 'random'")

    cluster_labels = np.unique(labels)
    n_clusters = int(cluster_labels.shape[0])
    colors = list(cc.glasbey_light[:max(n_clusters, 1)])

    fig_h = max(12, n_clusters)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=300)

    seg_h = 1.0   # 各クラスタの縦領域を等高に
    y_lower = 0.0
    yticks: List[float] = []
    yticklabels: List[str] = []

    rng = None
    if subsample_method == "random":
        rng = np.random.default_rng(seed)

    for i, c in enumerate(cluster_labels):
        vals = silhouette_samples[labels == c]
        vals = np.sort(vals)
        n_c = int(vals.size)
        if n_c == 0:
            continue

        # --- サブサンプリング ---
        if n_c > max_bars_per_cluster:
            if subsample_method == "stride":
                idx = np.linspace(0, n_c - 1, max_bars_per_cluster).astype(int)
            else:  # "random"
                assert rng is not None
                idx = np.sort(rng.choice(n_c, size=max_bars_per_cluster, replace=False))
            draw_vals = vals[idx]
            draw_n = max_bars_per_cluster
        else:
            draw_vals = vals
            draw_n = n_c

        y_pos = y_lower + np.linspace(0.0, seg_h, draw_n, endpoint=False)
        bar_h = seg_h / max(draw_n, 1)

        ax.barh(
            y_pos,
            draw_vals,
            height=bar_h,
            edgecolor=None,
            linewidth=0,
            color=colors[i % len(colors)],
        )

        yticks.append(y_lower + 0.5 * seg_h)
        yticklabels.append(f"{int(c)}")
        y_lower += seg_h

    sil_mean = float(np.mean(silhouette_samples)) if silhouette_samples.size else 0.0
    ax.axvline(
        sil_mean, color="red", linestyle="--", linewidth=1.0, label=f"mean={sil_mean:.3g}"
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, n_clusters * seg_h)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Cluster ID")
    ax.set_xlabel("Silhouette coefficient (cosine)")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper left")

    plt.tight_layout()
    try:
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300)
        if show:
            plt.show()
    finally:
        plt.close(fig)


# =========================================================
# Silhouette 統計・ブートストラップ
# =========================================================
def _calc_moments(x: np.ndarray) -> Tuple[float, float, float, float]:
    """1次～4次の標本モーメント（平均・標準偏差・歪度・尖度）を計算。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.nan, np.nan, 0.0, 0.0
    mean = float(np.mean(x))
    std = float(np.std(x))
    if n < 3 or std == 0.0 or not np.isfinite(std):
        return mean, std, 0.0, 0.0
    z = (x - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return mean, std, skew, kurt


def _cluster_stats_exact(s: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
    """
    指標 s をラベル y でクラスタ別に厳密集計し、
    per-cluster 統計リストと macro 平均を返す。
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    y_unique, y_inv = np.unique(y, return_inverse=True)
    K = y_unique.size

    counts = np.bincount(y_inv, minlength=K).astype(np.int64)
    sum_by = np.bincount(y_inv, weights=s, minlength=K).astype(np.float64)
    sumsq_by = np.bincount(y_inv, weights=s * s, minlength=K).astype(np.float64)

    means = sum_by / np.maximum(counts, 1)
    var = np.maximum(sumsq_by / np.maximum(counts, 1) - means**2, 0.0)
    std = np.sqrt(var)
    neg_rate = (
        np.bincount(y_inv, weights=(s < 0.0).astype(np.float64), minlength=K)
        / np.maximum(counts, 1)
    )

    per_cluster: List[Dict[str, Any]] = []
    for k in range(K):
        v = s[y_inv == k]
        if v.size:
            q25, q50, q75 = np.percentile(v, [25, 50, 75])
        else:
            q25 = q50 = q75 = np.nan
        per_cluster.append(
            {
                "cluster": int(y_unique[k])
                if np.issubdtype(y_unique.dtype, np.integer)
                else str(y_unique[k]),
                "n": int(counts[k]),
                "mean": float(means[k]),
                "std": float(std[k]),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "neg_rate": float(neg_rate[k]),
            }
        )

    macro_mean = float(np.mean(means)) if K > 0 else float("nan")
    return per_cluster, macro_mean


def _seed_streams(seed: int, device: str = "cuda") -> torch.Generator:
    """
    numpy.SeedSequence を用いて torch.Generator を初期化。
    """
    ss = np.random.SeedSequence(int(seed))
    g_torch = torch.Generator(device=device)
    g_torch.manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1)))
    return g_torch


@torch.no_grad()
def _bootstrap_mean_ci95_gpu_exact(
    s: np.ndarray,
    n_boot: int = 5000,
    device: str = "cuda",
    data_chunk: int = 5_000_000,
    boot_chunk: int = 100,
    dtype: torch.dtype = torch.float32,
    alpha: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, float]:
    """
    標準ブートストラップ平均の 95% CI（percentile 法）を GPU 上で計算。
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が利用できません。GPU 前提コードのため実行を停止します。")

    x_cpu = np.asarray(s, dtype=np.float32)
    n = int(x_cpu.size)
    if n == 0:
        return float("nan"), float("nan")

    x = torch.from_numpy(x_cpu).to(device, dtype=dtype, non_blocking=True)
    means_all = torch.empty(n_boot, dtype=dtype, device="cpu")

    n_full_chunks = n // data_chunk
    last_chunk = n - n_full_chunks * data_chunk

    done = 0
    g = generator if generator is not None else torch.Generator(device=device)

    while done < n_boot:
        b = min(boot_chunk, n_boot - done)
        acc = torch.zeros(b, device=device, dtype=dtype)

        for _ in range(n_full_chunks):
            idx = torch.randint(low=0, high=n, size=(b, data_chunk), device=device, generator=g)
            vals = x[idx]
            acc += vals.sum(dim=1)
            del idx, vals

        if last_chunk > 0:
            idx = torch.randint(low=0, high=n, size=(b, last_chunk), device=device, generator=g)
            vals = x[idx]
            acc += vals.sum(dim=1)
            del idx, vals

        means_all[done : done + b] = (acc / float(n)).to("cpu")
        done += b

    lo = float(torch.quantile(means_all, q=alpha / 2))
    hi = float(torch.quantile(means_all, q=1.0 - alpha / 2))
    return lo, hi


def compute_silhouette_report(
    silhouette_samples: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
    device: str = "cuda",
    data_chunk: int = 5_000_000,
    boot_chunk: int = 100,
) -> Dict[str, Any]:
    """
    Silhouette の全体/クラスタ別統計および平均の95% CIを計算し、JSON互換の辞書で返す。
    """
    g_torch = _seed_streams(seed=seed, device=device)

    s = np.asarray(silhouette_samples, dtype=np.float64)
    y = np.asarray(labels)

    n = int(s.size)
    k = int(np.unique(y).size)

    mean_s, std_s, skewness, kurtosis = _calc_moments(s)
    neg_rate = float(np.mean(s < 0.0)) if n else 0.0
    if n:
        q05, q25, q50, q75, q95 = np.percentile(s, [5, 25, 50, 75, 95])
    else:
        q05 = q25 = q50 = q75 = q95 = np.nan

    lo, hi = _bootstrap_mean_ci95_gpu_exact(
        s,
        n_boot=n_boot,
        device=device,
        data_chunk=data_chunk,
        boot_chunk=boot_chunk,
        generator=g_torch,
    )
    ci_method = f"bootstrap-gpu-exact(percentile; data_chunk={data_chunk}, boot_chunk={boot_chunk})"

    per_cluster, macro_mean = _cluster_stats_exact(s, y)
    weak_clusters = [c for c in per_cluster if (c["mean"] < 0.1 or c["neg_rate"] > 0.5)]

    summary_text = (
        f"Overall mean={mean_s:.3g} (95% CI: {lo:.3g}–{hi:.3g}, {ci_method}); "
        f"neg_rate={neg_rate:.3g}, skew={skewness:.3g}, kurt={kurtosis:.3g}. "
        f"Macro mean={macro_mean:.3g}. Weak clusters={len(weak_clusters)}."
    )

    return {
        "meta": {
            "n_samples": n,
            "n_clusters": k,
            "metric": "cosine (d = 1 - cos)",
            "ci_method": ci_method,
            "boot": {
                "n_boot": int(n_boot),
                "seed": int(seed),
                "device": device,
                "data_chunk": int(data_chunk),
                "boot_chunk": int(boot_chunk),
            },
        },
        "overall": {
            "mean": float(mean_s),
            "std": float(std_s),
            "ci95_mean": [float(lo), float(hi)],
            "neg_rate": float(neg_rate),
            "quantiles": {
                "q05": float(q05),
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
                "q95": float(q95),
            },
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
        },
        "macro_mean": float(macro_mean),
        "per_cluster": per_cluster,
        "weak_clusters": weak_clusters,
        "summary_text": summary_text,
    }


def build_diff_report(rep_ref: Dict[str, Any], rep_lat: Dict[str, Any]) -> Dict[str, Any]:
    """
    整合済みクラスタ ID を前提に、ref と latent の Silhouette 統計差分を作成。
    """
    idx_ref = {int(c["cluster"]): c for c in rep_ref["per_cluster"]}
    idx_lat = {int(c["cluster"]): c for c in rep_lat["per_cluster"]}
    clusters = sorted(set(idx_ref.keys()) | set(idx_lat.keys()))
    per_pair: List[Dict[str, Any]] = []

    for k in clusters:
        r = idx_ref.get(k)
        l = idx_lat.get(k)
        if (r is None) or (l is None):
            continue
        per_pair.append(
            {
                "cluster": k,
                "n_ref": r["n"],
                "n_lat": l["n"],
                "mean_ref": r["mean"],
                "mean_lat": l["mean"],
                "Δmean": l["mean"] - r["mean"],
                "neg_ref": r["neg_rate"],
                "neg_lat": l["neg_rate"],
                "Δneg": l["neg_rate"] - r["neg_rate"],
                "median_ref": r["median"],
                "median_lat": l["median"],
                "q25_ref": r["q25"],
                "q25_lat": l["q25"],
                "q75_ref": r["q75"],
                "q75_lat": l["q75"],
            }
        )

    overall = {
        "overall_mean_ref": rep_ref["overall"]["mean"],
        "overall_mean_lat": rep_lat["overall"]["mean"],
        "Δ_overall_mean": rep_lat["overall"]["mean"] - rep_ref["overall"]["mean"],
        "macro_mean_ref": rep_ref["macro_mean"],
        "macro_mean_lat": rep_lat["macro_mean"],
        "Δ_macro_mean": rep_lat["macro_mean"] - rep_ref["macro_mean"],
        "neg_rate_ref": rep_ref["overall"]["neg_rate"],
        "neg_rate_lat": rep_lat["overall"]["neg_rate"],
        "Δ_neg_rate": rep_lat["overall"]["neg_rate"] - rep_ref["overall"]["neg_rate"],
    }
    return {"overall": overall, "per_cluster": per_pair}


# =========================================================
# 保存ユーティリティ
# =========================================================
def save_report(report: Dict[str, Any], path: str | Path) -> None:
    """
    レポート辞書を JSON として保存する。
    ※ パスは呼び出し側で完全に指定する（eval_utils 内では固定しない）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to: {p}")


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """
    任意の辞書オブジェクトを JSON 形式で保存する。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Saved: {p}")

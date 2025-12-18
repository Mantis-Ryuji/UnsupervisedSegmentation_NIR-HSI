from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


PathLike = Union[str, Path]


def _safe_acos(x: torch.Tensor) -> torch.Tensor:
    r"""数値安定化付き arccos."""
    return torch.acos(torch.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7))


def sphere_log(mu: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Log 写像（球面 → 接空間）。

    Parameters
    ----------
    mu : torch.Tensor of shape (D,)
        単位球面 S^{D-1} 上の基準点。
    z : torch.Tensor of shape (..., D)
        単位球面上の点。
    eps : float, default=1e-8
        数値安定化用の微小値。

    Returns
    -------
    v : torch.Tensor of shape (..., D)
        mu における接空間 T_mu S^{D-1} 上のベクトル。

    Notes
    -----
    数学的には以下を計算している：

    - c = <mu, z>
    - theta = arccos(c)
    - u = z - c * mu
    - log_mu(z) = (theta / ||u||) * u

    z ≈ mu の場合は極限として 0 ベクトルを返す。
    """
    c = (z * mu).sum(dim=-1, keepdim=True)        # (..., 1)
    theta = _safe_acos(c)                         # (..., 1)
    u = z - c * mu                                # (..., D)
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    v = (theta / u_norm) * u
    v = torch.where(theta < 1e-6, torch.zeros_like(v), v)
    return v


def sphere_exp(mu: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Exp 写像（接空間 → 球面）。

    Parameters
    ----------
    mu : torch.Tensor of shape (D,)
        単位球面 S^{D-1} 上の基準点。
    v : torch.Tensor of shape (..., D)
        接空間ベクトル。
    eps : float, default=1e-8
        数値安定化用の微小値。

    Returns
    -------
    z : torch.Tensor of shape (..., D)
        単位球面上の点。

    Notes
    -----
    数学的には以下を計算している：

    - theta = ||v||
    - exp_mu(v) = cos(theta) * mu + sin(theta) * (v / theta)

    theta → 0 の場合は exp_mu(0) = mu とする。
    """
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)  # (..., 1)
    v_dir = v / theta.clamp_min(eps)
    z = torch.cos(theta) * mu + torch.sin(theta) * v_dir
    z = torch.where(theta < 1e-6, mu.expand_as(z), z)
    return F.normalize(z, dim=-1)


@torch.no_grad()
def frechet_mean_sphere(
    z: torch.Tensor,
    *,
    iters: int = 10,
    step: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    球面 S^{D-1} 上の Fréchet 平均（近似）を計算する。

    Parameters
    ----------
    z : torch.Tensor of shape (N, D)
        単位球面上の点群。
    iters : int, default=10
        Log/Exp による反復回数。
    step : float, default=1.0
        更新ステップ係数。
    eps : float, default=1e-8
        数値安定化用の微小値。

    Returns
    -------
    mu : torch.Tensor of shape (D,)
        単位球面上の Fréchet 平均（近似）。

    Notes
    -----
    分散が小さい場合は normalize(mean(z)) とほぼ一致する。
    """
    mu = F.normalize(z.mean(dim=0), dim=0)
    for _ in range(iters):
        v = sphere_log(mu, z, eps=eps)
        v_bar = v.mean(dim=0)
        mu = sphere_exp(mu, step * v_bar, eps=eps)
    return mu


class SphericalPGA1D:
    r"""
    球面 Principal Geodesic Analysis (PGA) による 1 次元進行度推定器。

    本クラスは、L2 正規化された潜在表現
    z ∈ S^{D-1}（例：ChemoMAE の latent）に対して、
    球面上の主測地線（principal geodesic）を推定し、
    各サンプルを 1 次元スカラー（進行度）に射影する。

    ユークリッド PCA との違い
    -------------------------
    - PCA : ユークリッド空間での直線方向
    - PGA : 球面多様体上での測地線（大円）方向

    劣化や経時変化のような「方向として表現された連続変化」を
    1D 軸として抽出する用途を想定している。

    Parameters
    ----------
    center_in_tangent : bool, default=True
        Log 写像後の接空間ベクトルを平均 0 にセンタリングしてから PCA を行う。
        True の場合、fit 時の平均 v_mean_ を保存し、
        transform / inverse_transform でも同一の補正を適用する。
    mean_iters : int, default=10
        Fréchet 平均推定の反復回数。
        0 の場合は normalize(mean(z)) を使用する。
    mean_step : float, default=1.0
        Fréchet 平均更新のステップ係数。
    eps : float, default=1e-8
        数値安定化用の微小値。

    Attributes
    ----------
    mu_ : torch.Tensor of shape (D,)
        球面平均（Fréchet 平均の近似）。
    p_ : torch.Tensor of shape (D,)
        接空間における主方向（mu_ と直交、unit norm）。
    v_mean_ : torch.Tensor of shape (D,)
        接空間ベクトルの平均（center_in_tangent=True の場合）。
    t_min_ : float
        学習データ上での t の最小値。
    t_max_ : float
        学習データ上での t の最大値。
    explained_var_ratio_ : float
        接空間 PCA における第 1 成分の分散説明率。
    is_fitted_ : bool
        学習済みかどうか。

    Notes
    -----
    - transform() が返す t はラジアン相当のスカラーであり、
      劣化進行度や擬似時間として解釈できる。
    - normalize=True を指定すると、fit 時に保存した範囲に基づき
      t を [0, 1] に正規化した u を返す。
    - inverse_transform() では、center_in_tangent=True の場合、
      v_mean_ を足し戻した上で Exp 写像を行うため、
      主測地線はデータ群の中心を正しく通過する。
    """

    def __init__(
        self,
        *,
        center_in_tangent: bool = True,
        mean_iters: int = 10,
        mean_step: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        self.center_in_tangent = bool(center_in_tangent)
        self.mean_iters = int(mean_iters)
        self.mean_step = float(mean_step)
        self.eps = float(eps)

        self.mu_: Optional[torch.Tensor] = None
        self.p_: Optional[torch.Tensor] = None
        self.v_mean_: Optional[torch.Tensor] = None

        # min-max range for t (learned on fit data)
        self.t_min_: Optional[float] = None
        self.t_max_: Optional[float] = None

        self.explained_var_ratio_: float = 0.0
        self.is_fitted_: bool = False

    @staticmethod
    def _check_z(z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor")
        if z.ndim != 2:
            raise ValueError(f"z must be (N,D), got shape={tuple(z.shape)}")
        if z.size(0) < 2:
            raise ValueError("z must contain at least 2 samples")
        return F.normalize(z, dim=1)

    def _check_fitted(self) -> None:
        if (not self.is_fitted_) or self.mu_ is None or self.p_ is None or self.v_mean_ is None:
            raise RuntimeError("Call fit() before using this method.")

    @torch.no_grad()
    def fit(self, z: torch.Tensor, *, mu: Optional[torch.Tensor] = None) -> "SphericalPGA1D":
        """
        球面PGAの主方向を学習し、学習データ上の t の min/max も保存する。
        """
        z = self._check_z(z)
        device, dtype = z.device, z.dtype

        # mean direction mu
        if mu is None:
            if self.mean_iters <= 0:
                mu_ = F.normalize(z.mean(dim=0), dim=0)
            else:
                mu_ = frechet_mean_sphere(z, iters=self.mean_iters, step=self.mean_step, eps=self.eps)
        else:
            if mu.ndim != 1 or mu.shape[0] != z.shape[1]:
                raise ValueError(f"mu must be (D,), got shape={tuple(mu.shape)}")
            mu_ = F.normalize(mu.to(device=device, dtype=dtype), dim=0)

        # tangent vectors
        V = sphere_log(mu_, z, eps=self.eps)  # (N,D)

        if self.center_in_tangent:
            v_mean_ = V.mean(dim=0)
            Vc = V - v_mean_[None, :]
        else:
            v_mean_ = torch.zeros_like(mu_)
            Vc = V

        # PCA in tangent space (SVD)
        U, S, Wt = torch.linalg.svd(Vc, full_matrices=False)
        p_ = Wt[0]  # (D,)
        p_ = p_ - (p_ @ mu_) * mu_
        p_ = F.normalize(p_, dim=0)

        var_total = (S**2).sum().item()
        var_1d = (S[0]**2).item() if S.numel() > 0 else 0.0
        evr = (var_1d / var_total) if var_total > 0 else 0.0

        # compute t on training data and store range
        t_train = (Vc @ p_)  # (N,)
        t_min = float(t_train.min().item())
        t_max = float(t_train.max().item())

        self.mu_ = mu_
        self.p_ = p_
        self.v_mean_ = v_mean_
        self.explained_var_ratio_ = float(evr)
        self.t_min_ = t_min
        self.t_max_ = t_max
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        fit() で保存した t_min_/t_max_ を用いて t を [0,1] に min-max 正規化する。
        """
        self._check_fitted()
        if self.t_min_ is None or self.t_max_ is None:
            raise RuntimeError("t_min_/t_max_ are not set. Call fit() first.")
        denom = (self.t_max_ - self.t_min_)
        if denom <= 1e-12:
            # 退化：全て同一t（極端に分散がない）
            return torch.zeros_like(t)
        return (t - self.t_min_) / denom

    @torch.no_grad()
    def denormalize_t(self, u: torch.Tensor) -> torch.Tensor:
        """
        u in [0,1] を、fit() 時のスケールの t に戻す。
        """
        self._check_fitted()
        if self.t_min_ is None or self.t_max_ is None:
            raise RuntimeError("t_min_/t_max_ are not set. Call fit() first.")
        return u * (self.t_max_ - self.t_min_) + self.t_min_

    @torch.no_grad()
    def transform(
        self,
        z: torch.Tensor,
        *,
        normalize: bool = False,
        return_delta: bool = False,
        return_z_hat: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        z を 1D 座標へ射影する。

        Parameters
        ----------
        normalize : bool, default=False
            True の場合、fit() 時の t_min_/t_max_ に基づき [0,1] に正規化した u を返す。
        return_z_hat : bool
            True の場合、主測地線上の再投影点 z_hat を返す（v_mean を足し戻した版）。
        return_delta : bool
            True の場合、delta = arccos(<z, z_hat>) を返す。

        Returns
        -------
        t_or_u : torch.Tensor
            normalize=False なら t（ラジアン相当）、True なら u∈[0,1]。
        """
        self._check_fitted()
        z = self._check_z(z).to(device=self.mu_.device, dtype=self.mu_.dtype)

        V = sphere_log(self.mu_, z, eps=self.eps)
        if self.center_in_tangent:
            Vc = V - self.v_mean_[None, :]
        else:
            Vc = V

        t = (Vc @ self.p_)  # (N,)
        t_or_u = self.normalize_t(t) if normalize else t

        outs: Tuple[torch.Tensor, ...] = (t_or_u,)
        if return_z_hat or return_delta:
            z_hat = self.inverse_transform(t_or_u, normalized=normalize)
            if return_z_hat:
                outs = outs + (z_hat,)
            if return_delta:
                delta = _safe_acos((z * z_hat).sum(dim=1))
                outs = outs + (delta,)
        return outs[0] if len(outs) == 1 else outs

    @torch.no_grad()
    def inverse_transform(self, t_or_u: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
        """
        1D 座標から主測地線上の点 z_hat を生成する。

        Parameters
        ----------
        t_or_u : torch.Tensor
            normalized=False の場合 t（(N,)）。normalized=True の場合 u∈[0,1]（(N,)）。
        normalized : bool, default=False
            True の場合、u を t に戻してから生成する。

        Returns
        -------
        z_hat : torch.Tensor
            (N,D) unit ベクトル。z_hat = exp_mu(v_mean + t p)（center_in_tangent=True の場合）。
        """
        self._check_fitted()

        if not isinstance(t_or_u, torch.Tensor):
            t_or_u = torch.as_tensor(t_or_u, device=self.mu_.device, dtype=self.mu_.dtype)
        else:
            t_or_u = t_or_u.to(device=self.mu_.device, dtype=self.mu_.dtype)

        if t_or_u.ndim == 0:
            t_or_u = t_or_u[None]
        if t_or_u.ndim != 1:
            raise ValueError(f"t_or_u must be 1D (N,), got shape={tuple(t_or_u.shape)}")

        t = self.denormalize_t(t_or_u) if normalized else t_or_u

        if self.center_in_tangent:
            V_hat = self.v_mean_[None, :] + t[:, None] * self.p_[None, :]
        else:
            V_hat = t[:, None] * self.p_[None, :]

        return sphere_exp(self.mu_, V_hat, eps=self.eps)

    def state_dict(self) -> Dict[str, Any]:
        """
        torch.save 用の状態辞書を返す。
        """
        self._check_fitted()
        return {
            "mu": self.mu_.detach().cpu(),
            "p": self.p_.detach().cpu(),
            "v_mean": self.v_mean_.detach().cpu(),
            "explained_var_ratio": float(self.explained_var_ratio_),
            "t_min": float(self.t_min_) if self.t_min_ is not None else None,
            "t_max": float(self.t_max_) if self.t_max_ is not None else None,
            "config": {
                "center_in_tangent": self.center_in_tangent,
                "mean_iters": self.mean_iters,
                "mean_step": self.mean_step,
                "eps": self.eps,
            },
        }

    def save(self, path: PathLike) -> None:
        """
        学習済みモデルを保存する（torch.save）。
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: PathLike, *, map_location: Optional[Union[str, torch.device]] = "cpu") -> "SphericalPGA1D":
        """
        保存済みモデルを復元する。
        """
        obj = torch.load(Path(path), map_location=map_location)
        cfg = obj.get("config", {})
        model = cls(
            center_in_tangent=cfg.get("center_in_tangent", True),
            mean_iters=cfg.get("mean_iters", 10),
            mean_step=cfg.get("mean_step", 1.0),
            eps=cfg.get("eps", 1e-8),
        )

        mu = obj["mu"]
        p = obj["p"]
        v_mean = obj["v_mean"]

        if mu.ndim != 1 or p.ndim != 1 or v_mean.ndim != 1:
            raise ValueError("Saved tensors must be 1D vectors (D,).")
        if not (mu.shape == p.shape == v_mean.shape):
            raise ValueError(f"Shape mismatch: mu={mu.shape}, p={p.shape}, v_mean={v_mean.shape}")

        model.mu_ = F.normalize(mu.to(dtype=torch.float32), dim=0)
        p_ = p.to(dtype=torch.float32)
        p_ = p_ - (p_ @ model.mu_) * model.mu_
        model.p_ = F.normalize(p_, dim=0)
        model.v_mean_ = v_mean.to(dtype=torch.float32)

        model.explained_var_ratio_ = float(obj.get("explained_var_ratio", 0.0))
        model.t_min_ = obj.get("t_min", None)
        model.t_max_ = obj.get("t_max", None)
        model.is_fitted_ = True
        return model
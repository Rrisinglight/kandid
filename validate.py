#!/usr/bin/env python3
"""
Валидация синтетических данных испытаний лазерного наведения с БПЛА.

Скрипт читает Excel-файлы, сгенерированные generate.py, и проверяет,
что данные перед построением графиков в analysis.ipynb имеют нужный
статистический рисунок. Каждый тест печатает явный PASS/FAIL.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_VALIDATION_DIR = SCRIPT_DIR / "validation"

ALPHA = 0.05
R2_MIN = 0.76
OMEGA_VISIBLE_MIN = 0.025
# Доля остатка в ANOVA type II немного консервативнее обычного R², поэтому
# держим её как отдельный мягкий порог, а не как строгое 1 - R2_MIN.
RESIDUAL_SS_SHARE_MAX = 0.26
VIF_MAX = 5.0
SHAPIRO_W_MIN = 0.90
DW_MIN = 1.35
DW_MAX = 2.5
SIGMA_RATIO_MIN = 0.25
SIGMA_RATIO_MAX = 1.60
SIGMA_CELL_PASS_MIN = 0.80
MIN_SIGMA_CELL_N = 20
LOW_HEIGHT_GALE_THRESHOLD = 3.0
LOW_HEIGHT_GALE_BOOST = 0.80

COL_MAP = {
    "Время": "time",
    "Точка": "point",
    "H (м)": "H",
    "Полёт": "flight",
    "V (м/с)": "V",
    "Порывы, откр. источник (м/с)": "gusts",
    "Направление ветра": "wind_dir",
    "Облачность (%)": "cloud",
    "Спутники": "sat",
    "r_x (мм)": "r_x",
    "r_y (мм)": "r_y",
    "R (мм)": "R",
}

FACTOR_ORDER = ["H", "V", "sat", "cloud"]
FACTOR_LABELS = {
    "C(H_factor)": "H",
    "H": "H",
    "V": "V",
    "sat": "sat",
    "cloud": "cloud",
}

# Та же VH-таблица, что в generate.py. Значения заданы в сантиметрах,
# validate.py переводит их в миллиметры и сравнивает с фактическим разбросом R.
VH_HEIGHTS = np.array([3.0, 5.0, 10.0, 15.0, 20.0])
VH_WIND_MIDS = np.array([1.5, 2.55, 3.25, 4.0, 6.1])
VH_SIGMA_CM = np.array([
    [4.0, 5.5, 8.0, 9.0, 19.0],
    [5.0, 7.5, 9.0, 10.5, 22.0],
    [7.0, 9.5, 11.0, 14.0, 25.0],
    [10.5, 11.0, 13.5, 14.5, 29.0],
    [13.0, 14.5, 15.5, 19.0, 36.0],
])


def low_height_gale_multiplier(h: float, v: float) -> float:
    """Та же поправка малых высот при сильном ветре, что и в generate.py."""
    if v <= LOW_HEIGHT_GALE_THRESHOLD or h >= 10.0:
        return 1.0

    height_weight = (10.0 - h) / (10.0 - VH_HEIGHTS[0])
    height_weight = float(np.clip(height_weight, 0.0, 1.0))
    gale_span = max(VH_WIND_MIDS[-1] - LOW_HEIGHT_GALE_THRESHOLD, 0.1)
    gale_excess = max(0.0, (v - LOW_HEIGHT_GALE_THRESHOLD) / gale_span)

    return 1.0 + LOW_HEIGHT_GALE_BOOST * height_weight * gale_excess ** 1.2


@dataclass
class Check:
    label: str
    ok: bool
    details: str = ""


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_check(label: str, ok: bool, details: str = "") -> Check:
    suffix = f" — {details}" if details else ""
    print(f"  [{status(ok)}] {label}{suffix}")
    return Check(label, ok, details)


def _interp1d_clamp(x: float, xs: np.ndarray) -> tuple[int, int, float]:
    if x <= xs[0]:
        return 0, 0, 0.0
    if x >= xs[-1]:
        return len(xs) - 1, len(xs) - 1, 0.0
    idx = int(np.searchsorted(xs, x, side="right")) - 1
    idx = min(idx, len(xs) - 2)
    t = (x - xs[idx]) / (xs[idx + 1] - xs[idx])
    return idx, idx + 1, float(t)


def reference_sigma_mm(h: float, v: float) -> float:
    hi0, hi1, ht = _interp1d_clamp(float(h), VH_HEIGHTS)
    vi0, vi1, vt = _interp1d_clamp(float(v), VH_WIND_MIDS)
    c00 = VH_SIGMA_CM[hi0, vi0]
    c01 = VH_SIGMA_CM[hi0, vi1]
    c10 = VH_SIGMA_CM[hi1, vi0]
    c11 = VH_SIGMA_CM[hi1, vi1]
    sigma_cm = (
        c00 * (1 - ht) * (1 - vt)
        + c01 * (1 - ht) * vt
        + c10 * ht * (1 - vt)
        + c11 * ht * vt
    )
    return float(sigma_cm) * 6.0 * low_height_gale_multiplier(h, v)


# ═══════════════════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════════════════

def load_measurements(input_dir: Path) -> pd.DataFrame:
    """Читает все Excel-файлы и собирает единый DataFrame."""
    frames: list[pd.DataFrame] = []
    files = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"Нет xlsx-файлов в {input_dir}")

    for fpath in files:
        df = pd.read_excel(fpath, header=12, engine="openpyxl")
        df.rename(columns=COL_MAP, inplace=True)
        df["date"] = fpath.stem
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    for col in ("H", "V", "cloud", "sat", "R", "r_x", "r_y", "gusts", "wind_dir"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result.dropna(subset=["H", "V", "R"], inplace=True)
    result = result[result["R"] > 0].copy()
    result["H_factor"] = result["H"].astype(int).astype(str)
    return result


def test_structure(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("0. СТРУКТУРА ДАННЫХ")
    print("=" * 68)

    checks: list[Check] = []
    required = set(COL_MAP.values()) | {"date"}
    missing = sorted(required - set(df.columns))
    checks.append(print_check("Все обязательные столбцы есть", not missing,
                              "missing=" + ", ".join(missing) if missing else ""))

    expected_heights = {3, 5, 10, 15, 20}
    actual_heights = {int(h) for h in df["H"].dropna().unique()}
    checks.append(print_check("Высоты 3, 5, 10, 15, 20 м присутствуют",
                              expected_heights.issubset(actual_heights),
                              f"H={sorted(actual_heights)}"))

    if "point" in df:
        points = sorted(str(p) for p in df["point"].dropna().unique())
        checks.append(print_check("Есть 5 контрольных точек", len(points) == 5,
                                  "points=" + ", ".join(points)))

    if "flight" in df:
        repeats = sorted(int(x) for x in df["flight"].dropna().unique())
        checks.append(print_check("Есть 3 повтора", repeats == [1, 2, 3],
                                  "flight=" + ", ".join(map(str, repeats))))

    print()
    return {"checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 1. Mixed model
# ═══════════════════════════════════════════════════════════════════════

def test_mixed_model(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("1. MIXED LINEAR MODEL  (R ~ H + V + sat + cloud | date)")
    print("=" * 68)

    used_fallback = False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = mixedlm("R ~ H + V + sat + cloud", data=df, groups=df["date"])
            result = md.fit(reml=True)
        print(result.summary())
        coefs = result.fe_params
        pvals = result.pvalues
    except Exception as exc:
        used_fallback = True
        print(f"  Mixed model не сошёлся — fallback на OLS: {exc}")
        result = ols("R ~ H + V + sat + cloud", data=df).fit()
        print(result.summary())
        coefs = result.params
        pvals = result.pvalues

    checks = [print_check("Модель построена", result is not None,
                          "OLS fallback" if used_fallback else "mixedlm")]
    print()
    return {
        "model": result,
        "coefs": coefs.to_dict(),
        "pvalues": pvals.to_dict(),
        "used_fallback": used_fallback,
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Wald / F-test — значимость факторов
# ═══════════════════════════════════════════════════════════════════════

def _p_verdict(p: float) -> str:
    if p < 0.001:
        return "СИЛЬНО значим (p < 0.001)"
    if p < 0.01:
        return "значим (p < 0.01)"
    if p < 0.05:
        return "слабо значим (p < 0.05)"
    if p < 0.1:
        return "тенденция (p < 0.1)"
    return "НЕ значим"


def test_factor_significance(mixed_result: dict) -> dict:
    pvals = mixed_result["pvalues"]
    print("=" * 68)
    print("2. WALD TEST — значимость факторов")
    print("=" * 68)

    verdicts: dict[str, str] = {}
    checks: list[Check] = []
    for factor in FACTOR_ORDER:
        p = float(pvals.get(factor, 1.0))
        verdict = _p_verdict(p)
        verdicts[factor] = verdict
        print(f"  {factor:6s}: p = {p:.4e}  →  {verdict}")

    print()
    checks.append(print_check("H сильно значим", float(pvals.get("H", 1.0)) < 0.001,
                              f"p={float(pvals.get('H', 1.0)):.2e}"))
    checks.append(print_check("V значим", float(pvals.get("V", 1.0)) < 0.05,
                              f"p={float(pvals.get('V', 1.0)):.2e}"))
    checks.append(print_check("sat видим статистически", float(pvals.get("sat", 1.0)) < 0.05,
                              f"p={float(pvals.get('sat', 1.0)):.2e}"))
    checks.append(print_check("cloud видим статистически", float(pvals.get("cloud", 1.0)) < 0.10,
                              f"p={float(pvals.get('cloud', 1.0)):.2e}"))
    print()
    return {"verdicts": verdicts, "pvalues": pvals, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 3. ANOVA II — η², ω², доли SS
# ═══════════════════════════════════════════════════════════════════════

def test_anova_effects(df: pd.DataFrame) -> dict:
    model = ols("R ~ C(H_factor) + V + sat + cloud", data=df).fit()
    aov = anova_lm(model, typ=2)

    ss_resid = float(aov.loc["Residual", "sum_sq"])
    df_resid = float(aov.loc["Residual", "df"])
    ms_resid = ss_resid / df_resid
    n = len(df)
    ss_total_type2 = float(aov["sum_sq"].sum())

    print("=" * 68)
    print("3. ANOVA II — сила факторов, ω² и доли SS")
    print("=" * 68)

    omega2: dict[str, float] = {}
    eta2: dict[str, float] = {}
    ss_share: dict[str, float] = {}
    rows: list[dict] = []

    for key in ("C(H_factor)", "V", "sat", "cloud"):
        if key not in aov.index:
            continue
        label = FACTOR_LABELS[key]
        ss_f = float(aov.loc[key, "sum_sq"])
        df_f = float(aov.loc[key, "df"])
        f_val = float(aov.loc[key, "F"])
        p_val = float(aov.loc[key, "PR(>F)"])
        e2 = ss_f / (ss_f + ss_resid)
        w2 = max(0.0, (ss_f - df_f * ms_resid) / (ss_f + (n - df_f) * ms_resid))
        share = ss_f / ss_total_type2
        eta2[label] = e2
        omega2[label] = w2
        ss_share[label] = share
        rows.append({"factor": label, "ss": ss_f, "df": df_f, "F": f_val,
                     "p": p_val, "eta2": e2, "omega2": w2, "ss_share": share})

    ss_share["residual"] = ss_resid / ss_total_type2

    for row in rows:
        bar = "█" * int(row["omega2"] * 80)
        print(f"  {row['factor']:6s}: SS={row['ss']:12.0f}  "
              f"share={row['ss_share'] * 100:5.1f}%  F={row['F']:8.1f}  "
              f"p={row['p']:.2e}  η²={row['eta2']:.4f}  ω²={row['omega2']:.4f} {bar}")
    print(f"  {'resid':6s}: SS={ss_resid:12.0f}  share={ss_share['residual'] * 100:5.1f}%")

    order = [k for k, _ in sorted(omega2.items(), key=lambda x: x[1], reverse=True)]
    print(f"\n  Порядок ω²: {' > '.join(order)}")
    print(f"  R² модели:            {model.rsquared:.4f}")
    print(f"  R² скорректированный: {model.rsquared_adj:.4f}")
    print()

    checks = [
        print_check("Иерархия ω²: H > V > sat > cloud", order == FACTOR_ORDER,
                    " > ".join(order)),
        print_check(f"sat виден на графике ω² >= {OMEGA_VISIBLE_MIN:.3f}",
                    omega2.get("sat", 0.0) >= OMEGA_VISIBLE_MIN,
                    f"ω²={omega2.get('sat', 0.0):.4f}"),
        print_check(f"cloud виден на графике ω² >= {OMEGA_VISIBLE_MIN:.3f}",
                    omega2.get("cloud", 0.0) >= OMEGA_VISIBLE_MIN,
                    f"ω²={omega2.get('cloud', 0.0):.4f}"),
        print_check(f"R² >= {R2_MIN:.2f}", model.rsquared >= R2_MIN,
                    f"R²={model.rsquared:.4f}"),
        print_check(f"Необъяснённая доля SS <= {RESIDUAL_SS_SHARE_MAX * 100:.0f}%",
                    ss_share["residual"] <= RESIDUAL_SS_SHARE_MAX,
                    f"residual={ss_share['residual'] * 100:.1f}%"),
    ]
    print()

    return {
        "model": model,
        "aov": aov,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "eta2": eta2,
        "omega2": omega2,
        "ss_share": ss_share,
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Проверка сильного ветра на малых высотах
# ═══════════════════════════════════════════════════════════════════════

def test_low_height_gale(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("4. СИЛЬНЫЙ ВЕТЕР НА МАЛЫХ ВЫСОТАХ")
    print("=" * 68)

    checks: list[Check] = []
    low = df[df["H"].isin([3, 5])].copy()
    calm = low[low["V"] < 3.0]["R"]
    gale = low[low["V"] >= 3.6]["R"]
    transition = low[(low["V"] >= 3.6) & (low["V"] <= 4.4)]["R"]

    median_ratio = float(gale.median() / calm.median()) if len(calm) and len(gale) else np.nan
    std_ratio = float(transition.std(ddof=1) / calm.std(ddof=1)) if len(calm) > 1 and len(transition) > 1 else np.nan
    if len(calm) and len(gale):
        mw_stat, mw_p = stats.mannwhitneyu(gale, calm, alternative="greater")
    else:
        mw_stat, mw_p = np.nan, np.nan

    print(f"  H=3/5 м, V<3.0:      n={len(calm):4d}, median(R)={calm.median():7.1f} мм, std={calm.std(ddof=1):7.1f} мм")
    print(f"  H=3/5 м, V>=3.6:     n={len(gale):4d}, median(R)={gale.median():7.1f} мм")
    print(f"  H=3/5 м, V=3.6–4.4:  n={len(transition):4d}, std(R)={transition.std(ddof=1):7.1f} мм")
    print(f"  Mann–Whitney: U={mw_stat:.0f}, p={mw_p:.4e}")

    h3_high = df[(df["H"] == 3) & (df["V"] >= 4.4)].copy()
    if len(h3_high) > 1:
        h3_std = float(h3_high["R"].std(ddof=1))
        h3_ref = float(np.mean([reference_sigma_mm(row.H, row.V) for row in h3_high.itertuples()]))
        h3_ratio = h3_std / h3_ref if h3_ref > 0 else np.nan
    else:
        h3_std = h3_ref = h3_ratio = np.nan
    print(f"  H=3 м, V>=4.4:       n={len(h3_high):4d}, std(R)={h3_std:7.1f} мм, "
          f"sigma_ref={h3_ref:7.1f} мм, ratio={h3_ratio:.2f}")

    checks.extend([
        print_check("Есть спокойные и ветровые наблюдения на H=3/5 м",
                    len(calm) >= MIN_SIGMA_CELL_N and len(gale) >= MIN_SIGMA_CELL_N,
                    f"calm={len(calm)}, gale={len(gale)}"),
        print_check("median(R) на H=3/5 м растёт при V >= 3.6 минимум на 25%",
                    np.isfinite(median_ratio) and median_ratio >= 1.25,
                    f"ratio={median_ratio:.2f}"),
        print_check("std(R) при V=3.6–4.4 выше спокойного режима минимум в 1.20 раза",
                    np.isfinite(std_ratio) and std_ratio >= 1.20,
                    f"ratio={std_ratio:.2f}"),
        print_check("H=3 м, V>=4.4 согласуется с reference sigma",
                    np.isfinite(h3_ratio) and 0.70 <= h3_ratio <= 1.60,
                    f"ratio={h3_ratio:.2f}"),
        print_check("Mann–Whitney подтверждает рост R при сильном ветре на H=3/5 м",
                    np.isfinite(mw_p) and mw_p < 0.001,
                    f"p={mw_p:.2e}"),
    ])
    print()
    return {
        "median_ratio": median_ratio,
        "std_ratio": std_ratio,
        "h3_sigma_ratio": h3_ratio,
        "mannwhitney_p": float(mw_p) if np.isfinite(mw_p) else np.nan,
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Brown–Forsythe / Levene — гетероскедастичность
# ═══════════════════════════════════════════════════════════════════════

def test_heteroscedasticity(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("5. BROWN–FORSYTHE / LEVENE — разброс R")
    print("=" * 68)

    results: dict[str, tuple[float, float]] = {}
    checks: list[Check] = []

    groups_h = [grp["R"].values for _, grp in df.groupby("H")]
    stat_h, p_h = stats.levene(*groups_h, center="median")
    results["H"] = (float(stat_h), float(p_h))
    print(f"  По H: stat={stat_h:.2f}, p={p_h:.4e}")
    checks.append(print_check("Разброс R различается по H", p_h < ALPHA,
                              f"p={p_h:.2e}"))

    df_tmp = df.copy()
    df_tmp["V_q"] = pd.qcut(df_tmp["V"], 4, labels=False, duplicates="drop")
    groups_v = [grp["R"].values for _, grp in df_tmp.groupby("V_q")]
    stat_v, p_v = stats.levene(*groups_v, center="median")
    results["V"] = (float(stat_v), float(p_v))
    print(f"  По V-квартилям: stat={stat_v:.2f}, p={p_v:.4e}")
    checks.append(print_check("Разброс R различается по V", p_v < ALPHA,
                              f"p={p_v:.2e}"))

    log_groups = [np.log(grp[grp > 0]) for grp in groups_h]
    stat_log, p_log = stats.levene(*log_groups, center="median")
    results["log_H"] = (float(stat_log), float(p_log))
    print(f"  По H для ln(R): stat={stat_log:.2f}, p={p_log:.4e}")
    checks.append(print_check("Лог-разброс не разваливается", stat_log < stat_h,
                              f"F_ln={stat_log:.2f} < F_R={stat_h:.2f}"))
    print()
    return {"results": results, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 6. Shapiro–Wilk + QQ-plot
# ═══════════════════════════════════════════════════════════════════════

def test_normality(df: pd.DataFrame, output_dir: Path) -> dict:
    model = ols("R ~ H + V + sat + H:V", data=df).fit()
    residuals = model.resid

    print("=" * 68)
    print("6. НОРМАЛЬНОСТЬ И АВТОКОРРЕЛЯЦИЯ ОСТАТКОВ")
    print("=" * 68)

    sample = residuals
    if len(residuals) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(residuals), 5000, replace=False)
        sample = residuals.iloc[idx]

    stat_sw, p_sw = stats.shapiro(sample)
    dw = float(durbin_watson(residuals))
    print(f"  Shapiro–Wilk: W={stat_sw:.4f}, p={p_sw:.4e}")
    print(f"  Durbin–Watson: DW={dw:.3f}")

    if plt is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("QQ-plot остатков OLS (R ~ H + V + sat + H:V)")
        ax.grid(True, alpha=0.3)
        qq_path = output_dir / "qq_plot.png"
        fig.savefig(qq_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  QQ-plot сохранён: {qq_path}")
    else:
        print("  QQ-plot не сохранён: matplotlib не установлен")

    checks = [
        print_check(f"Остатки достаточно похожи на нормальные: W >= {SHAPIRO_W_MIN:.2f}",
                    stat_sw >= SHAPIRO_W_MIN, f"W={stat_sw:.4f}"),
        print_check(f"Durbin–Watson в мягком диапазоне {DW_MIN:.1f}–{DW_MAX:.1f}",
                    DW_MIN <= dw <= DW_MAX, f"DW={dw:.3f}"),
    ]
    print()
    return {"shapiro_stat": float(stat_sw), "shapiro_p": float(p_sw),
            "durbin_watson": dw, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 7. Spearman — монотонная связь
# ═══════════════════════════════════════════════════════════════════════

def test_spearman(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("7. SPEARMAN — монотонная связь R с факторами")
    print("=" * 68)

    results: dict[str, tuple[float, float]] = {}
    for factor in FACTOR_ORDER:
        rho, p = stats.spearmanr(df[factor], df["R"])
        results[factor] = (float(rho), float(p))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  R ~ {factor:6s}: ρ = {rho:+.4f}, p = {p:.4e} {sig}")

    print()
    checks = [
        print_check("R монотонно растёт с H", results["H"][0] > 0 and results["H"][1] < ALPHA,
                    f"rho={results['H'][0]:+.3f}, p={results['H'][1]:.2e}"),
        print_check("R монотонно растёт с V", results["V"][0] > 0 and results["V"][1] < ALPHA,
                    f"rho={results['V'][0]:+.3f}, p={results['V'][1]:.2e}"),
        print_check("Больше спутников снижает R", results["sat"][0] < 0 and results["sat"][1] < ALPHA,
                    f"rho={results['sat'][0]:+.3f}, p={results['sat'][1]:.2e}"),
    ]
    print()
    return {"results": results, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 8. VIF — мультиколлинеарность
# ═══════════════════════════════════════════════════════════════════════

def test_vif(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("8. VIF — мультиколлинеарность предикторов")
    print("=" * 68)

    predictors = df[["H", "V", "sat", "cloud"]].copy()
    predictors["const"] = 1.0

    vif_values: dict[str, float] = {}
    for i, name in enumerate(FACTOR_ORDER):
        vif = float(variance_inflation_factor(predictors.values, i))
        vif_values[name] = vif
        flag = " ⚠" if vif >= VIF_MAX else ""
        print(f"  {name:6s}: VIF = {vif:.2f}{flag}")

    checks = [print_check(f"Все VIF < {VIF_MAX:.0f}", all(v < VIF_MAX for v in vif_values.values()),
                          ", ".join(f"{k}={v:.2f}" for k, v in vif_values.items()))]
    print()
    return {"values": vif_values, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# 9. Проверка фактического разброса против VH sigma
# ═══════════════════════════════════════════════════════════════════════

def test_sigma_reference(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("9. VH-ТАБЛИЦА — фактический разброс R против reference sigma")
    print("=" * 68)

    wind_edges = [0.0]
    wind_edges.extend(((VH_WIND_MIDS[:-1] + VH_WIND_MIDS[1:]) / 2).tolist())
    wind_edges.append(np.inf)
    df_tmp = df.copy()
    df_tmp["V_ref"] = pd.cut(df_tmp["V"], bins=wind_edges, labels=VH_WIND_MIDS)

    rows: list[dict] = []
    for (h, v_ref), grp in df_tmp.dropna(subset=["V_ref"]).groupby(["H", "V_ref"], observed=True):
        if len(grp) < MIN_SIGMA_CELL_N:
            continue
        ref_sigma = reference_sigma_mm(float(h), float(v_ref))
        actual_std = float(grp["R"].std(ddof=1))
        ratio = actual_std / ref_sigma if ref_sigma > 0 else np.nan
        ok = SIGMA_RATIO_MIN <= ratio <= SIGMA_RATIO_MAX
        rows.append({
            "H": float(h), "V_ref": float(v_ref), "n": len(grp),
            "std_R": actual_std, "sigma_ref": ref_sigma, "ratio": ratio, "ok": ok,
        })

    for row in rows:
        print(f"  H={row['H']:>4.0f} м, V≈{row['V_ref']:>4.2f} м/с, "
              f"n={row['n']:4d}: std(R)={row['std_R']:7.1f} мм, "
              f"sigma_ref={row['sigma_ref']:7.1f} мм, ratio={row['ratio']:.2f} "
              f"[{status(row['ok'])}]")

    pass_share = float(np.mean([r["ok"] for r in rows])) if rows else 0.0
    median_ratio = float(np.median([r["ratio"] for r in rows])) if rows else np.nan
    checks = [
        print_check("Есть достаточно VH-ячеек для проверки", bool(rows), f"cells={len(rows)}"),
        print_check(f"Не меньше {SIGMA_CELL_PASS_MIN * 100:.0f}% VH-ячеек в мягком диапазоне "
                    f"{SIGMA_RATIO_MIN:.2f}–{SIGMA_RATIO_MAX:.2f}",
                    pass_share >= SIGMA_CELL_PASS_MIN,
                    f"pass={pass_share * 100:.1f}%, median ratio={median_ratio:.2f}"),
    ]
    print()
    return {"rows": rows, "pass_share": pass_share, "median_ratio": median_ratio, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# Диагностические графики
# ═══════════════════════════════════════════════════════════════════════

def plot_diagnostics(df: pd.DataFrame, anova: dict, output_dir: Path) -> None:
    if plt is None:
        print("Диагностические графики не сохранены: matplotlib не установлен")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    ax = axes[0, 0]
    heights = sorted(df["H"].unique())
    data_by_h = [df[df["H"] == h]["R"].values for h in heights]
    ax.boxplot(data_by_h, tick_labels=[str(int(h)) for h in heights], whis=1.5)
    ax.set_xlabel("H (м)")
    ax.set_ylabel("R (мм)")
    ax.set_title("R vs H")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(df["V"], df["R"], alpha=0.15, s=4, color="steelblue")
    z = np.polyfit(df["V"], df["R"], 1)
    v_line = np.linspace(df["V"].min(), df["V"].max(), 100)
    ax.plot(v_line, np.polyval(z, v_line), "r-", linewidth=2)
    ax.set_xlabel("V (м/с)")
    ax.set_ylabel("R (мм)")
    ax.set_title("R vs V")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    model = ols("R ~ H + V + sat + H:V", data=df).fit()
    ax.scatter(model.fittedvalues, model.resid, alpha=0.15, s=4, color="coral")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    omega = anova["omega2"]
    names = FACTOR_ORDER
    vals = [omega.get(name, 0.0) for name in names]
    ax.bar(names, vals, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    ax.axhline(OMEGA_VISIBLE_MIN, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Сила влияния факторов (ω²)")
    ax.set_ylabel("ω²")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    corr_cols = ["H", "V", "sat", "cloud", "R"]
    corr = df[corr_cols].corr(method="spearman")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45)
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.set_title("Spearman correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    ax.hist(df["R"], bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("R (мм)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of R")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    df_tmp = df.copy()
    df_tmp["sat_q"] = pd.qcut(df_tmp["sat"], 4, labels=False, duplicates="drop")
    sat_groups = sorted(df_tmp["sat_q"].dropna().unique())
    data_by_sat = [df_tmp[df_tmp["sat_q"] == q]["R"].values for q in sat_groups]
    labels = []
    for q in sat_groups:
        subset = df_tmp[df_tmp["sat_q"] == q]["sat"]
        labels.append(f"{subset.min():.0f}-{subset.max():.0f}")
    ax.boxplot(data_by_sat, tick_labels=labels)
    ax.set_xlabel("Спутники (квартили)")
    ax.set_ylabel("R (мм)")
    ax.set_title("R vs sat")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    ss_share = anova["ss_share"]
    labels = ["H", "V", "sat", "cloud", "residual"]
    vals = [ss_share.get(name, 0.0) for name in labels]
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#bdc3c7"])
    ax.set_title("Доли SS: факторы и остаток")

    fig.suptitle("Диагностика синтетических данных", fontsize=14, y=1.01)
    fig.tight_layout()
    diag_path = output_dir / "diagnostics.png"
    fig.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Диагностические графики: {diag_path}")


# ═══════════════════════════════════════════════════════════════════════
# Сводный отчёт
# ═══════════════════════════════════════════════════════════════════════

def build_summary(*sections: dict) -> list[str]:
    checks: list[Check] = []
    for section in sections:
        checks.extend(section.get("checks", []))

    lines = ["СВОДКА ВАЛИДАЦИИ", "=" * 40, ""]
    for chk in checks:
        suffix = f" — {chk.details}" if chk.details else ""
        lines.append(f"  [{status(chk.ok)}] {chk.label}{suffix}")

    passed = sum(1 for chk in checks if chk.ok)
    total = len(checks)
    lines.append("")
    lines.append(f"Итого: {passed}/{total} проверок пройдено")
    return lines


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Валидация синтетических данных лазерного наведения с БПЛА",
    )
    p.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help="Папка с Excel-файлами от generate.py",
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_VALIDATION_DIR),
        help="Папка для отчётов и графиков",
    )
    p.add_argument(
        "--fail-exit-code", action="store_true",
        help="Вернуть код 1, если есть хотя бы один FAIL",
    )
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка данных из {input_dir}...")
    df = load_measurements(input_dir)
    print(f"Загружено: {len(df)} измерений, {df['date'].nunique()} дней\n")

    print("─" * 68)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print("─" * 68)
    for col in ("H", "V", "sat", "cloud", "R"):
        s = df[col]
        print(f"  {col:6s}: mean={s.mean():8.2f}, std={s.std():8.2f}, "
              f"min={s.min():8.2f}, max={s.max():8.2f}")
    print()

    structure = test_structure(df)
    mixed = test_mixed_model(df)
    significance = test_factor_significance(mixed)
    anova = test_anova_effects(df)
    low_height_gale = test_low_height_gale(df)
    hetero = test_heteroscedasticity(df)
    normality = test_normality(df, output_dir)
    spearman = test_spearman(df)
    vif = test_vif(df)
    sigma_ref = test_sigma_reference(df)

    plot_diagnostics(df, anova, output_dir)

    summary = build_summary(
        structure, mixed, significance, anova, low_height_gale,
        hetero, normality, spearman, vif, sigma_ref,
    )
    print()
    for line in summary:
        print(line)

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")

    if args.fail_exit_code and any("[FAIL]" in line for line in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

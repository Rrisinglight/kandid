#!/usr/bin/env python3
"""
Валидация синтетических данных испытаний лазерного наведения с БПЛА.

Читает Excel-файлы, сгенерированные generate.py, и запускает набор
статистических тестов по критериям из Main.md:
  1. Mixed model   — основная проверка
  2. Wald / F-test — значимость факторов
  3. ω² (omega²)   — сила эффектов
  4. Brown–Forsythe / Levene — гетероскедастичность
  5. Shapiro–Wilk + QQ-plot — нормальность остатков
  6. Spearman      — монотонный рост R с H и V
  7. VIF           — мультиколлинеарность предикторов

Ожидаемая иерархия:  H > V > sat > cloud
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_VALIDATION_DIR = SCRIPT_DIR / "validation"

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


# ═══════════════════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════════════════

def load_measurements(input_dir: Path) -> pd.DataFrame:
    """Читает все Excel-файлы и собирает единый DataFrame."""
    frames: list[pd.DataFrame] = []
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Нет xlsx-файлов в {input_dir}")

    for fpath in files:
        df = pd.read_excel(
            fpath,
            header=12,   # строка 13 (0-indexed: 12)
            engine="openpyxl",
        )
        df.rename(columns=COL_MAP, inplace=True)
        df["date"] = fpath.stem
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    for col in ("H", "V", "cloud", "sat", "R", "r_x", "r_y"):
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result.dropna(subset=["H", "V", "R"], inplace=True)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 1. Mixed model
# ═══════════════════════════════════════════════════════════════════════

def test_mixed_model(df: pd.DataFrame) -> dict:
    md = mixedlm("R ~ H + V + sat + cloud", data=df, groups=df["date"])
    result = md.fit(reml=True)
    print("=" * 68)
    print("1. MIXED LINEAR MODEL  (R ~ H + V + sat + cloud | date)")
    print("=" * 68)
    print(result.summary())
    print()

    coefs = result.fe_params
    pvals = result.pvalues
    return {
        "model": result,
        "coefs": coefs.to_dict(),
        "pvalues": pvals.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Wald / F-test — значимость факторов
# ═══════════════════════════════════════════════════════════════════════

def test_factor_significance(mixed_result: dict) -> dict:
    pvals = mixed_result["pvalues"]
    print("=" * 68)
    print("2. WALD TEST — значимость факторов")
    print("=" * 68)

    verdicts: dict[str, str] = {}
    for factor in ("H", "V", "sat", "cloud"):
        p = pvals.get(factor, 1.0)
        if p < 0.001:
            verdict = "СИЛЬНО значим (p < 0.001)"
        elif p < 0.01:
            verdict = "значим (p < 0.01)"
        elif p < 0.05:
            verdict = "слабо значим (p < 0.05)"
        elif p < 0.1:
            verdict = "тенденция (p < 0.1)"
        else:
            verdict = "НЕ значим"
        verdicts[factor] = verdict
        print(f"  {factor:6s}: p = {p:.4e}  →  {verdict}")

    print()
    return verdicts


# ═══════════════════════════════════════════════════════════════════════
# 3. Partial ω² (omega-squared)
# ═══════════════════════════════════════════════════════════════════════

def test_effect_sizes(df: pd.DataFrame) -> dict:
    model = ols("R ~ H + V + sat + cloud", data=df).fit()
    aov = anova_lm(model, typ=2)

    ss_resid = aov.loc["Residual", "sum_sq"]
    df_resid = aov.loc["Residual", "df"]
    ms_resid = ss_resid / df_resid
    n = len(df)

    print("=" * 68)
    print("3. EFFECT SIZES — partial ω²")
    print("=" * 68)

    omega2: dict[str, float] = {}
    for factor in ("H", "V", "sat", "cloud"):
        if factor not in aov.index:
            omega2[factor] = 0.0
            continue
        ss_f = aov.loc[factor, "sum_sq"]
        df_f = aov.loc[factor, "df"]
        w2 = (ss_f - df_f * ms_resid) / (ss_f + (n - df_f) * ms_resid)
        w2 = max(0.0, w2)
        omega2[factor] = w2

    sorted_effects = sorted(omega2.items(), key=lambda x: x[1], reverse=True)
    for factor, w2 in sorted_effects:
        bar = "█" * int(w2 * 200)
        print(f"  {factor:6s}: ω² = {w2:.4f}  {bar}")

    order = [f for f, _ in sorted_effects]
    expected = ["H", "V", "sat", "cloud"]
    match = order == expected
    print(f"\n  Порядок: {' > '.join(order)}")
    print(f"  Ожидаемый: {' > '.join(expected)}")
    print(f"  Совпадение: {'ДА' if match else 'НЕТ'}")
    print()
    return omega2


# ═══════════════════════════════════════════════════════════════════════
# 4. Brown–Forsythe / Levene — гетероскедастичность
# ═══════════════════════════════════════════════════════════════════════

def test_heteroscedasticity(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("4. BROWN–FORSYTHE / LEVENE — гетероскедастичность")
    print("=" * 68)

    results: dict[str, tuple[float, float]] = {}

    # По высотам
    groups_h = [grp["R"].values for _, grp in df.groupby("H")]
    stat_h, p_h = stats.levene(*groups_h, center="median")
    results["H"] = (stat_h, p_h)
    print(f"  По H: stat={stat_h:.2f}, p={p_h:.4e}  "
          f"→ {'Гетероскедастичность' if p_h < 0.05 else 'Гомоскедастичность'}")

    # По квартилям ветра
    df_tmp = df.copy()
    df_tmp["V_q"] = pd.qcut(df_tmp["V"], 4, labels=False, duplicates="drop")
    groups_v = [grp["R"].values for _, grp in df_tmp.groupby("V_q")]
    stat_v, p_v = stats.levene(*groups_v, center="median")
    results["V"] = (stat_v, p_v)
    print(f"  По V: stat={stat_v:.2f}, p={p_v:.4e}  "
          f"→ {'Гетероскедастичность' if p_v < 0.05 else 'Гомоскедастичность'}")

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. Shapiro–Wilk + QQ-plot
# ═══════════════════════════════════════════════════════════════════════

def test_normality(df: pd.DataFrame, output_dir: Path) -> dict:
    model = ols("R ~ H + V + sat + cloud", data=df).fit()
    residuals = model.resid

    print("=" * 68)
    print("5. SHAPIRO–WILK — нормальность остатков")
    print("=" * 68)

    # Shapiro-Wilk (на подвыборке если n > 5000)
    sample = residuals
    if len(residuals) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(residuals), 5000, replace=False)
        sample = residuals.iloc[idx]

    stat_sw, p_sw = stats.shapiro(sample)
    print(f"  Shapiro–Wilk: W={stat_sw:.4f}, p={p_sw:.4e}")

    dw = durbin_watson(residuals)
    print(f"  Durbin–Watson: DW={dw:.3f}")

    # QQ-plot
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("QQ-plot остатков OLS (R ~ H + V + sat + cloud)")
    ax.grid(True, alpha=0.3)
    qq_path = output_dir / "qq_plot.png"
    fig.savefig(qq_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  QQ-plot сохранён: {qq_path}")
    print()
    return {"shapiro_stat": stat_sw, "shapiro_p": p_sw, "durbin_watson": dw}


# ═══════════════════════════════════════════════════════════════════════
# 6. Spearman — монотонный рост
# ═══════════════════════════════════════════════════════════════════════

def test_spearman(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("6. SPEARMAN — монотонная связь R с факторами")
    print("=" * 68)

    results: dict[str, tuple[float, float]] = {}
    for factor in ("H", "V", "sat", "cloud"):
        rho, p = stats.spearmanr(df[factor], df["R"])
        results[factor] = (rho, p)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  R ~ {factor:6s}: ρ = {rho:+.4f}, p = {p:.4e} {sig}")

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
# 7. VIF — мультиколлинеарность
# ═══════════════════════════════════════════════════════════════════════

def test_vif(df: pd.DataFrame) -> dict:
    print("=" * 68)
    print("7. VIF — мультиколлинеарность предикторов")
    print("=" * 68)

    predictors = df[["H", "V", "sat", "cloud"]].copy()
    predictors["const"] = 1.0

    vif_values: dict[str, float] = {}
    names = ["H", "V", "sat", "cloud"]
    for i, name in enumerate(names):
        vif = variance_inflation_factor(predictors.values, i)
        vif_values[name] = vif
        flag = " ⚠" if vif > 5 else ""
        print(f"  {name:6s}: VIF = {vif:.2f}{flag}")

    print()
    return vif_values


# ═══════════════════════════════════════════════════════════════════════
# Диагностические графики
# ═══════════════════════════════════════════════════════════════════════

def plot_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. R vs H — boxplot
    ax = axes[0, 0]
    heights = sorted(df["H"].unique())
    data_by_h = [df[df["H"] == h]["R"].values for h in heights]
    ax.boxplot(data_by_h, tick_labels=[str(int(h)) for h in heights], whis=1.5)
    ax.set_xlabel("H (м)")
    ax.set_ylabel("R (мм)")
    ax.set_title("R vs H")
    ax.grid(True, alpha=0.3)

    # 2. R vs V — scatter + тренд
    ax = axes[0, 1]
    ax.scatter(df["V"], df["R"], alpha=0.15, s=4, color="steelblue")
    z = np.polyfit(df["V"], df["R"], 1)
    v_line = np.linspace(df["V"].min(), df["V"].max(), 100)
    ax.plot(v_line, np.polyval(z, v_line), "r-", linewidth=2)
    ax.set_xlabel("V (м/с)")
    ax.set_ylabel("R (мм)")
    ax.set_title("R vs V")
    ax.grid(True, alpha=0.3)

    # 3. Residuals vs fitted
    ax = axes[0, 2]
    model = ols("R ~ H + V + sat + cloud", data=df).fit()
    ax.scatter(model.fittedvalues, model.resid, alpha=0.15, s=4, color="coral")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    # 4. Корреляционная матрица
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
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.set_title("Spearman correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 5. Распределение R (histogram)
    ax = axes[1, 1]
    ax.hist(df["R"], bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("R (мм)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of R")
    ax.grid(True, alpha=0.3)

    # 6. R vs sat — boxplot по квартилям
    ax = axes[1, 2]
    df_tmp = df.copy()
    df_tmp["sat_q"] = pd.qcut(df_tmp["sat"], 4, labels=False, duplicates="drop")
    sat_groups = sorted(df_tmp["sat_q"].unique())
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

    fig.suptitle("Диагностика синтетических данных", fontsize=14, y=1.01)
    fig.tight_layout()
    diag_path = output_dir / "diagnostics.png"
    fig.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Диагностические графики: {diag_path}")


# ═══════════════════════════════════════════════════════════════════════
# Сводный отчёт
# ═══════════════════════════════════════════════════════════════════════

def build_summary(
    verdicts: dict[str, str],
    omega2: dict[str, float],
    hetero: dict[str, tuple[float, float]],
    normality: dict,
    spearman: dict[str, tuple[float, float]],
    vif: dict[str, float],
) -> list[str]:
    lines = ["СВОДКА ВАЛИДАЦИИ", "=" * 40, ""]

    checks: list[tuple[str, bool]] = []

    # Иерархия ω²
    sorted_w = sorted(omega2.items(), key=lambda x: x[1], reverse=True)
    order = [f for f, _ in sorted_w]
    hierarchy_ok = order == ["H", "V", "sat", "cloud"]
    checks.append(("Иерархия ω²: H > V > sat > cloud", hierarchy_ok))

    # H и V — сильно значимы
    h_sig = "СИЛЬНО" in verdicts.get("H", "")
    v_sig = "значим" in verdicts.get("V", "").lower()
    checks.append(("H — сильно значим", h_sig))
    checks.append(("V — значим", v_sig))

    # Гетероскедастичность по H
    het_h = hetero.get("H", (0, 1))
    checks.append(("Гетероскедастичность по H", het_h[1] < 0.05))

    # Гетероскедастичность по V
    het_v = hetero.get("V", (0, 1))
    checks.append(("Гетероскедастичность по V", het_v[1] < 0.05))

    # Spearman R~H > 0
    rho_h = spearman.get("H", (0, 1))
    checks.append(("Spearman R~H > 0", rho_h[0] > 0 and rho_h[1] < 0.05))

    # Spearman R~V > 0
    rho_v = spearman.get("V", (0, 1))
    checks.append(("Spearman R~V > 0", rho_v[0] > 0 and rho_v[1] < 0.05))

    # VIF < 5
    all_vif_ok = all(v < 5 for v in vif.values())
    checks.append(("Все VIF < 5", all_vif_ok))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {label}")

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

    # Описательная статистика
    print("─" * 68)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print("─" * 68)
    for col in ("H", "V", "sat", "cloud", "R"):
        s = df[col]
        print(f"  {col:6s}: mean={s.mean():8.2f}, std={s.std():8.2f}, "
              f"min={s.min():8.2f}, max={s.max():8.2f}")
    print()

    mixed = test_mixed_model(df)
    verdicts = test_factor_significance(mixed)
    omega2 = test_effect_sizes(df)
    hetero = test_heteroscedasticity(df)
    normality = test_normality(df, output_dir)
    spearman = test_spearman(df)
    vif = test_vif(df)

    plot_diagnostics(df, output_dir)

    summary = build_summary(verdicts, omega2, hetero, normality, spearman, vif)
    print()
    for line in summary:
        print(line)

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()

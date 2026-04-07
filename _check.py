import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.formula.api import ols

frames = []
for f in sorted(Path("output").glob("*.xlsx")):
    tmp = pd.read_excel(f, header=12, engine="openpyxl")
    tmp.columns = ["time","point","H","flight","V","gusts","wind_dir","cloud","sat","r_x","r_y","R"]
    frames.append(tmp)
df = pd.concat(frames, ignore_index=True)
for c in ["H","V","R","r_x","r_y","sat","cloud"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

m1 = ols("R ~ H + V + sat + H:V", data=df).fit()
print(f"Linear  R ~ H + V + sat + H:V  :  R2={m1.rsquared:.4f}  R2_adj={m1.rsquared_adj:.4f}")

df["logR"] = np.log(df["R"])
m2 = ols("logR ~ H + V + sat + H:V", data=df).fit()
print(f"Log     lnR ~ H + V + sat + H:V:  R2={m2.rsquared:.4f}  R2_adj={m2.rsquared_adj:.4f}")

print(f"\nR distribution:")
for h in [3, 5, 10, 15, 20]:
    s = df[df["H"]==h]
    lo = s[s["V"]<=3]["R"].mean()
    hi = s[s["V"]>=5]["R"].mean()
    pct = (s["R"] > 200).mean() * 100
    print(f"  H={h:2d}: mean={s['R'].mean():.0f}, V<=3: {lo:.0f}, V>=5: {hi:.0f}, >200mm: {pct:.1f}%")

near = ((df["r_x"].abs() < 10) & (df["r_y"].abs() < 10)).sum()
print(f"\nCenter hits (|r_x|<10 & |r_y|<10): {near}")
print(f"Min R: {df['R'].min():.2f}, Max R: {df['R'].max():.1f}")

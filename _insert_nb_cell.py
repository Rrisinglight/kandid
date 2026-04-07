"""Insert markdown cell after fig_17_coef_forest code cell."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent / "analysis.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Примечание к рис. 17.** Абсолютные величины оценок $\\hat\\beta$ **нельзя напрямую сравнивать** между предикторами: у $H$, $V$ и числа спутников **разные единицы измерения** (м, м/с, шт.), поэтому больший коэффициент у $V$ не означает, что ветер «сильнее высоты» в смысле вклада в дисперсию $R$. **Ранжирование силы факторов** в данной работе опирается на дисперсионный анализ и $\\omega^2$ (рис. 15), а не на модуль $\\hat\\beta$ на графике коэффициентов.\n"
    ],
}

insert_at = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "fig_17_coef_forest" in src and "ALL_FIGS.append" in src:
        insert_at = i + 1
        break

if insert_at is None:
    raise SystemExit("fig_17 cell not found")

nb["cells"].insert(insert_at, new_cell)
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("inserted at index", insert_at)

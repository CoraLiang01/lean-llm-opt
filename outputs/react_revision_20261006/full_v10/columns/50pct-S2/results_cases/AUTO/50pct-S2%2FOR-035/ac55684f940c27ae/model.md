Let $F$ be the set of all foods in cost.csv.

Decision variables:
For each food $f \in F$, let $x_f \geq 0$ denote the number of servings of food $f$ in the meal plan (continuous, may be fractional).

Parameters (from cost.csv, table_id: file_0_view_0):
For each $f \in F$,
- $cal_f$: Calories per serving (column "Calories")
- $prot_f$: Protein per serving in grams (column "Protein(g)")
- $fat_f$: Fat per serving in grams (column "Fat(g)")
- $vitc_f$: Vitamin C per serving in mg (column "VitaminC(mg)")
- $cost_f$: Cost per serving in USD (column "Cost")

Model:

Minimize total cost:
$$
\min \sum_{f \in F} cost_f \, x_f
$$

Subject to:
- Calorie requirement:
$$
\sum_{f \in F} cal_f \, x_f \geq 2000
$$

- Protein requirement:
$$
\sum_{f \in F} prot_f \, x_f \geq 50
$$

- Vitamin C requirement:
$$
\sum_{f \in F} vitc_f \, x_f \geq 60
$$

- Fat limit:
$$
\sum_{f \in F} fat_f \, x_f \leq 70
$$

- Non-negativity:
$$
x_f \geq 0 \quad \forall f \in F
$$

Data Mapping:
- $F$: All foods in column "Food" of file_0_view_0 (cost.csv)
- $cal_f$: "Calories" column, file_0_view_0
- $prot_f$: "Protein(g)" column, file_0_view_0
- $fat_f$: "Fat(g)" column, file_0_view_0
- $vitc_f$: "VitaminC(mg)" column, file_0_view_0
- $cost_f$: "Cost" column, file_0_view_0
- $x_f$: servings of food $f$ (continuous, $\geq 0$)

All constraints and variables are defined over the full set of foods in the current cost.csv.
**Mathematical Model**

Let $F$ be the set of all foods in the table (see Data Mapping for exact list and identifiers).

**Decision Variables:**
- $x_f \geq 0$: Number of servings (possibly fractional) of food $f \in F$ to include in the meal plan.

**Parameters (from Data Mapping):**
- $c_f$: Calories per serving of food $f$
- $p_f$: Protein (g) per serving of food $f$
- $a_f$: Fat (g) per serving of food $f$
- $v_f$: Vitamin C (mg) per serving of food $f$
- $cost_f$: Cost (USD) per serving of food $f$

**Objective:**
$$
\min \sum_{f \in F} cost_f \, x_f
$$

**Subject to:**
- Calorie requirement:
  $$
  \sum_{f \in F} c_f \, x_f \geq 2000
  $$
- Protein requirement:
  $$
  \sum_{f \in F} p_f \, x_f \geq 50
  $$
- Vitamin C requirement:
  $$
  \sum_{f \in F} v_f \, x_f \geq 60
  $$
- Fat limit:
  $$
  \sum_{f \in F} a_f \, x_f \leq 70
  $$
- Non-negativity:
  $$
  x_f \geq 0 \quad \forall f \in F
  $$

---

**Data Mapping**

- Index set $F$: All foods in cost.csv, column "Food", table_id: file_0_view_0
- $c_f$: "Calories" column, file_0_view_0
- $p_f$: "Protein(g)" column, file_0_view_0
- $a_f$: "Fat(g)" column, file_0_view_0
- $v_f$: "VitaminC(mg)" column, file_0_view_0
- $cost_f$: "Cost" column, file_0_view_0
- $x_f$: servings of food $f$ (continuous, $\geq 0$)

All foods in the table are eligible for selection. All constraints and parameters are directly mapped from the specified columns in cost.csv (table_id: file_0_view_0).
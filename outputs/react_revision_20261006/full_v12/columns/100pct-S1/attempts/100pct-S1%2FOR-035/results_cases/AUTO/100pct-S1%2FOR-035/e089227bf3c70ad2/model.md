Let $F$ be the set of all foods in cost.csv.

**Decision variables:**
- $x_f \geq 0$: number of servings of food $f \in F$ (can be fractional)

**Parameters (from cost.csv, for each $f \in F$):**
- $c_f$: Calories per serving
- $p_f$: Protein (g) per serving
- $fat_f$: Fat (g) per serving
- $v_f$: VitaminC (mg) per serving
- $cost_f$: Cost (USD) per serving

**Objective:**
\[
\min \sum_{f \in F} cost_f \, x_f
\]

**Subject to:**
\[
\sum_{f \in F} c_f \, x_f \geq 2000 \qquad \text{(Calories)}
\]
\[
\sum_{f \in F} p_f \, x_f \geq 50 \qquad \text{(Protein)}
\]
\[
\sum_{f \in F} v_f \, x_f \geq 60 \qquad \text{(Vitamin C)}
\]
\[
\sum_{f \in F} fat_f \, x_f \leq 70 \qquad \text{(Fat)}
\]
\[
x_f \geq 0 \qquad \forall f \in F
\]

**Data Mapping:**
- $F$: All rows in cost.csv, column "Food"
- $c_f$: column "Calories"
- $p_f$: column "Protein(g)"
- $fat_f$: column "Fat(g)"
- $v_f$: column "VitaminC(mg)"
- $cost_f$: column "Cost"
- $x_f$: servings of food $f$ (continuous, $\geq 0$)

**Source Table:** cost.csv (table_id: file_0_view_0)
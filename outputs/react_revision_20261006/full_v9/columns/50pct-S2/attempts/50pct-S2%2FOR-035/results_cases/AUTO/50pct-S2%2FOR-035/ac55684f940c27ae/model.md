## Mathematical Model

**Sets:**
- Let $\mathcal{F}$ be the set of all foods in cost.csv (Food column, table_id: file_0_view_0).

**Parameters (for each $f \in \mathcal{F}$, from table_id: file_0_view_0):**
- $c_f$: Calories per serving (Calories)
- $p_f$: Protein per serving (Protein(g))
- $a_f$: Fat per serving (Fat(g))
- $v_f$: Vitamin C per serving (VitaminC(mg))
- $q_f$: Cost per serving (Cost)

**Decision Variables:**
- $x_f \geq 0$: Number of servings of food $f$ (can be fractional)

**Objective:**
\[
\min \sum_{f \in \mathcal{F}} q_f x_f
\]

**Subject to:**
\[
\sum_{f \in \mathcal{F}} c_f x_f \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{f \in \mathcal{F}} p_f x_f \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{f \in \mathcal{F}} v_f x_f \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{f \in \mathcal{F}} a_f x_f \leq 70 \qquad \text{(Fat upper bound)}
\]
\[
x_f \geq 0 \qquad \forall f \in \mathcal{F}
\]

---

**Data Mapping:**

- $\mathcal{F}$: All rows in column Food, table_id: file_0_view_0
- $c_f$: Calories, table_id: file_0_view_0
- $p_f$: Protein(g), table_id: file_0_view_0
- $a_f$: Fat(g), table_id: file_0_view_0
- $v_f$: VitaminC(mg), table_id: file_0_view_0
- $q_f$: Cost, table_id: file_0_view_0

All constraints and variables are defined over the full set of foods in the current cost.csv. All parameter values are taken directly from the corresponding columns in the file. Portions $x_f$ are continuous and nonnegative.
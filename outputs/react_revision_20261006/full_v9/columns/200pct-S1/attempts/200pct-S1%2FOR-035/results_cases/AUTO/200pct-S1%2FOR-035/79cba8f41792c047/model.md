Mathematical Optimization Model

Sets:
- Let $\mathcal{F}$ be the set of all foods in cost.csv (Food column, table_id: file_0_view_0).

Parameters (for each $i \in \mathcal{F}$, from table_id: file_0_view_0):
- $c_i$: Calories per serving (Calories)
- $p_i$: Protein per serving (Protein(g))
- $f_i$: Fat per serving (Fat(g))
- $v_i$: Vitamin C per serving (VitaminC(mg))
- $cost_i$: Cost per serving (Cost)

Decision Variables:
- $x_i \geq 0$: Number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

Objective:
$$
\min \sum_{i \in \mathcal{F}} cost_i \, x_i
$$

Subject to:
\[
\begin{align*}
&\sum_{i \in \mathcal{F}} c_i \, x_i \geq 2000 &&\text{(Calories requirement)} \\
&\sum_{i \in \mathcal{F}} p_i \, x_i \geq 50 &&\text{(Protein requirement)} \\
&\sum_{i \in \mathcal{F}} v_i \, x_i \geq 60 &&\text{(Vitamin C requirement)} \\
&\sum_{i \in \mathcal{F}} f_i \, x_i \leq 70 &&\text{(Fat upper bound)} \\
&x_i \geq 0 \quad \forall i \in \mathcal{F} &&\text{(Nonnegativity, servings may be fractional)}
\end{align*}
\]

Data Mapping:
- $\mathcal{F}$: All rows in cost.csv, column Food, table_id: file_0_view_0
- $c_i$: Calories, table_id: file_0_view_0
- $p_i$: Protein(g), table_id: file_0_view_0
- $f_i$: Fat(g), table_id: file_0_view_0
- $v_i$: VitaminC(mg), table_id: file_0_view_0
- $cost_i$: Cost, table_id: file_0_view_0

All constraints and variables are defined over the full set of foods in the provided data. No food is excluded. All parameter values are taken directly from the corresponding columns in cost.csv.
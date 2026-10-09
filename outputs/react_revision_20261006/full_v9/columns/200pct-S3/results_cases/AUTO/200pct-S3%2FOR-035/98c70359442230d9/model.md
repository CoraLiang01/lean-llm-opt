Mathematical Optimization Model

Sets:
- Let $\mathcal{F}$ be the set of all foods in cost.csv (indexed by $i$).

Parameters (from cost.csv, table_id: file_0_view_0):
- $c_i$: Calories per serving of food $i$ (column: Calories)
- $p_i$: Protein (g) per serving of food $i$ (column: Protein(g))
- $f_i$: Fat (g) per serving of food $i$ (column: Fat(g))
- $v_i$: Vitamin C (mg) per serving of food $i$ (column: VitaminC(mg))
- $q_i$: Cost (USD) per serving of food $i$ (column: Cost)

Decision Variables:
- $x_i \geq 0$: Number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

Objective:
$$
\min \sum_{i \in \mathcal{F}} q_i x_i
$$

Subject to:
\[
\begin{align*}
&\sum_{i \in \mathcal{F}} c_i x_i \geq 2000 &&\text{(Calories requirement)} \\
&\sum_{i \in \mathcal{F}} p_i x_i \geq 50 &&\text{(Protein requirement)} \\
&\sum_{i \in \mathcal{F}} v_i x_i \geq 60 &&\text{(Vitamin C requirement)} \\
&\sum_{i \in \mathcal{F}} f_i x_i \leq 70 &&\text{(Fat upper bound)} \\
&x_i \geq 0 \quad \forall i \in \mathcal{F} &&\text{(Non-negativity, servings may be fractional)}
\end{align*}
\]

Data Mapping:
- Foods: All rows in cost.csv, table_id: file_0_view_0, column Food
- $c_i$: column Calories
- $p_i$: column Protein(g)
- $f_i$: column Fat(g)
- $v_i$: column VitaminC(mg)
- $q_i$: column Cost

All foods in the file are eligible for selection. All constraints and variables are as described above.
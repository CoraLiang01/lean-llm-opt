Mathematical Optimization Model

Index Sets:
- $F$: Set of foods (from column Food in file_0_view_0).

Parameters:
- $cal_f$: Calories per serving of food $f$ (Calories, file_0_view_0)
- $prot_f$: Protein per serving of food $f$ (Protein(g), file_0_view_0)
- $fat_f$: Fat per serving of food $f$ (Fat(g), file_0_view_0)
- $vitc_f$: Vitamin C per serving of food $f$ (VitaminC(mg), file_0_view_0)
- $cost_f$: Cost per serving of food $f$ (Cost, file_0_view_0)

Decision Variables:
- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, may be fractional)

Objective:
$$
\min \sum_{f \in F} cost_f \cdot x_f
$$

Subject to:
\[
\begin{align*}
&\sum_{f \in F} cal_f \cdot x_f \geq 2000 \\
&\sum_{f \in F} prot_f \cdot x_f \geq 50 \\
&\sum_{f \in F} vitc_f \cdot x_f \geq 60 \\
&\sum_{f \in F} fat_f \cdot x_f \leq 70 \\
&x_f \geq 0 \quad \forall f \in F
\end{align*}
\]

Data Mapping

- $F$: file_0_view_0, column Food
- $cal_f$: file_0_view_0, column Calories
- $prot_f$: file_0_view_0, column Protein(g)
- $fat_f$: file_0_view_0, column Fat(g)
- $vitc_f$: file_0_view_0, column VitaminC(mg)
- $cost_f$: file_0_view_0, column Cost
- $x_f$: servings of food $f$ (decision variable, indexed by Food from file_0_view_0)
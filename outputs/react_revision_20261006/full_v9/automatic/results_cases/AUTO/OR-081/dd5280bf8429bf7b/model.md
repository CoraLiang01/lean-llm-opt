Mathematical Model

Sets:
- $F$: Set of all foods in cost.csv (indexed by $i$)

Parameters (from cost.csv, table_id: file_0_view_0):
- $c_i$: Cost per serving of food $i$ (column: Cost)
- $cal_i$: Calories per serving of food $i$ (column: Calories)
- $prot_i$: Protein (g) per serving of food $i$ (column: Protein(g))
- $fat_i$: Fat (g) per serving of food $i$ (column: Fat(g))
- $vc_i$: Vitamin C (mg) per serving of food $i$ (column: VitaminC(mg))

Decision Variables:
- $x_i \geq 0$: Number of servings (possibly fractional) of food $i$ to include in the meal plan

Objective:
$$
\min \sum_{i \in F} c_i x_i
$$

Subject to:
- Calorie requirement:
$$
\sum_{i \in F} cal_i x_i \geq 2000
$$

- Protein requirement:
$$
\sum_{i \in F} prot_i x_i \geq 50
$$

- Vitamin C requirement:
$$
\sum_{i \in F} vc_i x_i \geq 60
$$

- Fat upper bound:
$$
\sum_{i \in F} fat_i x_i \leq 70
$$

- Nonnegativity:
$$
x_i \geq 0 \quad \forall i \in F
$$

Data Mapping:
- $F$: All foods in file_0_view_0, column Food
- $c_i$: file_0_view_0, column Cost
- $cal_i$: file_0_view_0, column Calories
- $prot_i$: file_0_view_0, column Protein(g)
- $fat_i$: file_0_view_0, column Fat(g)
- $vc_i$: file_0_view_0, column VitaminC(mg)
- $x_i$: servings of food $i$ (continuous, $\geq 0$)

All parameter values are taken directly from the corresponding columns in cost.csv (file_0_view_0). The model includes all 120 foods.
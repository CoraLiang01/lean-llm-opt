Let $F$ be the set of all foods in cost.csv.

Parameters (from file_0_view_0 columns):
For each $i \in F$:
- $c_i$: Calories per serving (Calories)
- $p_i$: Protein per serving (Protein(g))
- $f_i$: Fat per serving (Fat(g))
- $v_i$: Vitamin C per serving (VitaminC(mg))
- $C_i$: Cost per serving (Cost)

Decision variables:
For each $i \in F$:
- $x_i \geq 0$: number of servings of food $i$ (can be fractional)

Objective:
Minimize total cost:
$$
\min \sum_{i \in F} C_i x_i
$$

Subject to:
Calorie requirement:
$$
\sum_{i \in F} c_i x_i \geq 2000
$$

Protein requirement:
$$
\sum_{i \in F} p_i x_i \geq 50
$$

Vitamin C requirement:
$$
\sum_{i \in F} v_i x_i \geq 60
$$

Fat limit:
$$
\sum_{i \in F} f_i x_i \leq 70
$$

Nonnegativity:
$$
x_i \geq 0 \quad \forall i \in F
$$

Data Mapping:
- $F$: All foods in file_0_view_0["Food"]
- $c_i$: file_0_view_0["Calories"] for food $i$
- $p_i$: file_0_view_0["Protein(g)"] for food $i$
- $f_i$: file_0_view_0["Fat(g)"] for food $i$
- $v_i$: file_0_view_0["VitaminC(mg)"] for food $i$
- $C_i$: file_0_view_0["Cost"] for food $i$
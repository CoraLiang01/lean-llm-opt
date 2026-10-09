Let $F$ be the set of all foods in cost.csv.

**Sets:**
- $F$: Set of foods (indexed by $i$), as listed in cost.csv.

**Parameters (from cost.csv, table_id: file_0_view_0):**
- $c_i$: Calories per serving of food $i$
- $p_i$: Protein (g) per serving of food $i$
- $f_i$: Fat (g) per serving of food $i$
- $v_i$: Vitamin C (mg) per serving of food $i$
- $cost_i$: Cost (USD) per serving of food $i$

**Decision Variables:**
- $x_i \geq 0$: Number of servings (possibly fractional) of food $i$ to include in the meal plan

**Objective:**
\[
\min \sum_{i \in F} cost_i \, x_i
\]

**Subject to:**
\[
\sum_{i \in F} c_i \, x_i \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{i \in F} p_i \, x_i \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{i \in F} v_i \, x_i \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{i \in F} f_i \, x_i \leq 70 \qquad \text{(Fat upper bound)}
\]
\[
x_i \geq 0 \qquad \forall i \in F
\]

**Data Mapping:**
- $F$ = all rows in cost.csv, table_id: file_0_view_0, column "Food"
- $c_i$ = column "Calories" for food $i$
- $p_i$ = column "Protein(g)" for food $i$
- $f_i$ = column "Fat(g)" for food $i$
- $v_i$ = column "VitaminC(mg)" for food $i$
- $cost_i$ = column "Cost" for food $i$
- $x_i$ = servings of food $i$ (continuous, $\geq 0$)

All foods in the file are eligible. All constraints and variables are as above.
Let $F$ be the set of all foods in cost.csv.

**Sets:**
- $F$: Set of foods (indexed by $f$), as listed in cost.csv.

**Parameters (from cost.csv, for each $f \in F$):**
- $cal_f$: Calories per serving of food $f$
- $prot_f$: Protein (g) per serving of food $f$
- $fat_f$: Fat (g) per serving of food $f$
- $vitc_f$: Vitamin C (mg) per serving of food $f$
- $cost_f$: Cost (USD) per serving of food $f$

**Decision Variables:**
- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{f \in F} cost_f \, x_f
\]

**Subject to:**
\[
\sum_{f \in F} cal_f \, x_f \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{f \in F} prot_f \, x_f \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{f \in F} vitc_f \, x_f \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{f \in F} fat_f \, x_f \leq 70 \qquad \text{(Fat limit)}
\]
\[
x_f \geq 0 \qquad \forall f \in F
\]

**Data Mapping:**
- $F$ = all foods in cost.csv, table_id: file_0_view_0, column: Food
- $cal_f$ = Calories, table_id: file_0_view_0, column: Calories
- $prot_f$ = Protein(g), table_id: file_0_view_0, column: Protein(g)
- $fat_f$ = Fat(g), table_id: file_0_view_0, column: Fat(g)
- $vitc_f$ = VitaminC(mg), table_id: file_0_view_0, column: VitaminC(mg)
- $cost_f$ = Cost, table_id: file_0_view_0, column: Cost

**Summary:**  
Minimize total cost of servings $x_f$ of each food $f \in F$ to meet or exceed 2000 kcal, 50 g protein, 60 mg vitamin C, and not exceed 70 g fat, with $x_f \geq 0$ and all nutrient and cost data as specified in cost.csv.
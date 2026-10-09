**Mathematical Model**

**Sets:**
- $F$: Set of all foods in cost.csv (indexed by $i$).

**Parameters (from cost.csv, table_id: file_0_view_0):**
- $c_i$: Cost per serving of food $i$ (column: Cost)
- $cal_i$: Calories per serving of food $i$ (column: Calories)
- $prot_i$: Protein per serving of food $i$ (column: Protein(g))
- $fat_i$: Fat per serving of food $i$ (column: Fat(g))
- $vc_i$: Vitamin C per serving of food $i$ (column: VitaminC(mg))

**Decision Variables:**
- $x_i \geq 0$: Number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{i \in F} c_i x_i
\]

**Subject to:**
\[
\sum_{i \in F} cal_i\, x_i \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{i \in F} prot_i\, x_i \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{i \in F} vc_i\, x_i \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{i \in F} fat_i\, x_i \leq 70 \qquad \text{(Fat upper bound)}
\]
\[
x_i \geq 0 \qquad \forall i \in F
\]

**Data Mapping**
- $F$: All foods in cost.csv, table_id: file_0_view_0, column: Food
- $c_i$: Cost, column: Cost
- $cal_i$: Calories, column: Calories
- $prot_i$: Protein(g), column: Protein(g)
- $fat_i$: Fat(g), column: Fat(g)
- $vc_i$: VitaminC(mg), column: VitaminC(mg)
- $x_i$: servings of food $i$ (continuous, $\geq 0$)

**All parameters and sets are directly mapped from cost.csv, table_id: file_0_view_0, using the specified columns.**
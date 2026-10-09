### Sets
- $F$: set of all foods in cost.csv (indexed by $f$)

### Parameters (from cost.csv, table_id: file_0_view_0)
- $\text{cal}_f$: Calories per serving of food $f$ (column: Calories)
- $\text{prot}_f$: Protein (g) per serving of food $f$ (column: Protein(g))
- $\text{fat}_f$: Fat (g) per serving of food $f$ (column: Fat(g))
- $\text{vitC}_f$: VitaminC (mg) per serving of food $f$ (column: VitaminC(mg))
- $\text{cost}_f$: Cost (USD) per serving of food $f$ (column: Cost)

### Decision Variables
- $x_f \geq 0$: number of servings of food $f$ (can be fractional)

### Objective
Minimize total cost:
$$
\min \sum_{f \in F} \text{cost}_f \, x_f
$$

### Constraints

Calorie requirement:
$$
\sum_{f \in F} \text{cal}_f \, x_f \geq 2000
$$

Protein requirement:
$$
\sum_{f \in F} \text{prot}_f \, x_f \geq 50
$$

Vitamin C requirement:
$$
\sum_{f \in F} \text{vitC}_f \, x_f \geq 60
$$

Fat upper bound:
$$
\sum_{f \in F} \text{fat}_f \, x_f \leq 70
$$

Nonnegativity:
$$
x_f \geq 0 \quad \forall f \in F
$$

---

**Data Mapping:**  
- Foods, nutrients, and costs are from cost.csv (table_id: file_0_view_0), columns: Food, Calories, Protein(g), Fat(g), VitaminC(mg), Cost.
- All foods in the file are included in $F$.
- All parameters are mapped directly from their respective columns.
- All constraints and the objective use these parameters and the decision variables as defined above.
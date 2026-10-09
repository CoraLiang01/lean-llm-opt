### Optimization Model

**Sets:**
- $F$: Set of all foods in cost.csv (indexed by $f$)

**Parameters (from Data Mapping):**
- $cal_f$: Calories per serving of food $f$ (column "Calories", table_id: file_0_view_0)
- $prot_f$: Protein (g) per serving of food $f$ (column "Protein(g)", table_id: file_0_view_0)
- $fat_f$: Fat (g) per serving of food $f$ (column "Fat(g)", table_id: file_0_view_0)
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (column "VitaminC(mg)", table_id: file_0_view_0)
- $cost_f$: Cost (USD) per serving of food $f$ (column "Cost", table_id: file_0_view_0)

**Decision Variables:**
- $x_f \geq 0$: Number of servings (possibly fractional) of food $f$ to include in the meal plan

**Objective:**
\[
\min \sum_{f \in F} cost_f \cdot x_f
\]

**Subject to:**
\[
\sum_{f \in F} cal_f \cdot x_f \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{f \in F} prot_f \cdot x_f \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{f \in F} vitc_f \cdot x_f \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{f \in F} fat_f \cdot x_f \leq 70 \qquad \text{(Fat limit)}
\]
\[
x_f \geq 0 \qquad \forall f \in F
\]

---

**Data Mapping:**

- $F$: All foods in column "Food" of table_id: file_0_view_0 (cost.csv)
- $cal_f$: "Calories" column, table_id: file_0_view_0
- $prot_f$: "Protein(g)" column, table_id: file_0_view_0
- $fat_f$: "Fat(g)" column, table_id: file_0_view_0
- $vitc_f$: "VitaminC(mg)" column, table_id: file_0_view_0
- $cost_f$: "Cost" column, table_id: file_0_view_0
- $x_f$: servings of food $f$, continuous and nonnegative

**All constraints and variables are defined over the full set of foods in the current cost.csv.**
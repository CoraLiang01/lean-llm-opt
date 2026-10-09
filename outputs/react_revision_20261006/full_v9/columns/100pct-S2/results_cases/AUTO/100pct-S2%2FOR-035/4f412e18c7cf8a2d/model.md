**Mathematical Model**

**Sets:**
- $F$: Set of all foods in cost.csv (indexed by $i$)

**Parameters (from file_0_view_0):**
- $\text{cal}_i$: Calories per serving of food $i$ (column "Calories")
- $\text{prot}_i$: Protein (g) per serving of food $i$ (column "Protein(g)")
- $\text{fat}_i$: Fat (g) per serving of food $i$ (column "Fat(g)")
- $\text{vitc}_i$: Vitamin C (mg) per serving of food $i$ (column "VitaminC(mg)")
- $\text{cost}_i$: Cost (USD) per serving of food $i$ (column "Cost")

**Decision Variables:**
- $x_i \geq 0$: Number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{i \in F} \text{cost}_i \, x_i
\]

**Subject to:**
\[
\sum_{i \in F} \text{cal}_i \, x_i \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{i \in F} \text{prot}_i \, x_i \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{i \in F} \text{vitc}_i \, x_i \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{i \in F} \text{fat}_i \, x_i \leq 70 \qquad \text{(Fat limit)}
\]
\[
x_i \geq 0 \qquad \forall i \in F
\]

**Data Mapping**
- $F$: All foods in column "Food" of table_id file_0_view_0 (cost.csv)
- $\text{cal}_i$: "Calories" column, file_0_view_0, row $i$
- $\text{prot}_i$: "Protein(g)" column, file_0_view_0, row $i$
- $\text{fat}_i$: "Fat(g)" column, file_0_view_0, row $i$
- $\text{vitc}_i$: "VitaminC(mg)" column, file_0_view_0, row $i$
- $\text{cost}_i$: "Cost" column, file_0_view_0, row $i$
- $x_i$: servings of food $i$ (decision variable for each $i$ in $F$)

**All data is sourced from file_0_view_0 (cost.csv), columns as specified above.**
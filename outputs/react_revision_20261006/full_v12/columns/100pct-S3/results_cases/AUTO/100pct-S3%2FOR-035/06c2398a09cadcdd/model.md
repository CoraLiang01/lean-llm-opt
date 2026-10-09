Let $F$ be the set of all foods in cost.csv.

**Decision variables:**
- For each food $i \in F$, let $x_i \geq 0$ be the number of servings of food $i$ in the meal plan (continuous, may be fractional).

**Parameters (from cost.csv, for each $i \in F$):**
- $\text{cal}_i$: Calories per serving of food $i$ (column "Calories")
- $\text{prot}_i$: Protein per serving of food $i$ (column "Protein(g)")
- $\text{fat}_i$: Fat per serving of food $i$ (column "Fat(g)")
- $\text{vitc}_i$: Vitamin C per serving of food $i$ (column "VitaminC(mg)")
- $\text{cost}_i$: Cost per serving of food $i$ (column "Cost")

**Objective:**
\[
\min \sum_{i \in F} \text{cost}_i\, x_i
\]

**Subject to:**
\[
\sum_{i \in F} \text{cal}_i\, x_i \geq 2000
\]
\[
\sum_{i \in F} \text{prot}_i\, x_i \geq 50
\]
\[
\sum_{i \in F} \text{vitc}_i\, x_i \geq 60
\]
\[
\sum_{i \in F} \text{fat}_i\, x_i \leq 70
\]
\[
x_i \geq 0 \quad \forall i \in F
\]

**Data Mapping:**
- $F$: All foods in cost.csv, column "Food", table_id: file_0_view_0
- $\text{cal}_i$: "Calories", file_0_view_0
- $\text{prot}_i$: "Protein(g)", file_0_view_0
- $\text{fat}_i$: "Fat(g)", file_0_view_0
- $\text{vitc}_i$: "VitaminC(mg)", file_0_view_0
- $\text{cost}_i$: "Cost", file_0_view_0
- $x_i$: servings of food $i$ (continuous, $\geq 0$)

**Sets and Indices:**
- $i \in F$ indexes all foods in cost.csv

**Summary:**  
Minimize total cost of servings chosen from all foods, subject to minimum calorie, protein, and vitamin C requirements, and a maximum fat limit, using per-serving data from cost.csv. Portions may be fractional.
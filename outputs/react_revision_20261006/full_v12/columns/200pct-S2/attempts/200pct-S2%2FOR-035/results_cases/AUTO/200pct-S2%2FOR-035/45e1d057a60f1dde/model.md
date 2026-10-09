Let $F$ be the set of all foods in cost.csv.

**Decision variables:**
- $x_f \geq 0$: number of servings of food $f \in F$ (can be fractional)

**Parameters (from cost.csv, for each $f \in F$):**
- $\text{Calories}_f$: Calories per serving of food $f$
- $\text{Protein}_f$: Protein (g) per serving of food $f$
- $\text{Fat}_f$: Fat (g) per serving of food $f$
- $\text{VitaminC}_f$: Vitamin C (mg) per serving of food $f$
- $\text{Cost}_f$: Cost (USD) per serving of food $f$

**Objective:**
\[
\min \sum_{f \in F} \text{Cost}_f \, x_f
\]

**Subject to:**
\[
\sum_{f \in F} \text{Calories}_f \, x_f \geq 2000
\]
\[
\sum_{f \in F} \text{Protein}_f \, x_f \geq 50
\]
\[
\sum_{f \in F} \text{VitaminC}_f \, x_f \geq 60
\]
\[
\sum_{f \in F} \text{Fat}_f \, x_f \leq 70
\]
\[
x_f \geq 0 \quad \forall f \in F
\]

**Data Mapping:**
- $F$: All foods in cost.csv, column "Food", table_id: file_0_view_0
- $\text{Calories}_f$: column "Calories", table_id: file_0_view_0
- $\text{Protein}_f$: column "Protein(g)", table_id: file_0_view_0
- $\text{Fat}_f$: column "Fat(g)", table_id: file_0_view_0
- $\text{VitaminC}_f$: column "VitaminC(mg)", table_id: file_0_view_0
- $\text{Cost}_f$: column "Cost", table_id: file_0_view_0

**Summary:**  
Minimize total cost of servings $x_f$ of each food $f$, subject to meeting at least 2000 kcal, 50 g protein, 60 mg vitamin C, and at most 70 g fat, with $x_f \geq 0$ and all data as specified in cost.csv.
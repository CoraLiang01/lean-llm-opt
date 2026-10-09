### Mathematical Model

**Sets:**
- $F$: Set of all foods in cost.csv (see Data Mapping for exact list).

**Parameters (from cost.csv, indexed by $i \in F$):**
- $c_i$: Calories per serving of food $i$ (column "Calories")
- $p_i$: Protein (g) per serving of food $i$ (column "Protein(g)")
- $f_i$: Fat (g) per serving of food $i$ (column "Fat(g)")
- $v_i$: Vitamin C (mg) per serving of food $i$ (column "VitaminC(mg)")
- $cost_i$: Cost (USD) per serving of food $i$ (column "Cost")

**Decision Variables:**
- $x_i \geq 0$: Number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

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

---

**Data Mapping**

- Table: cost.csv (table_id: file_0_view_0)
    - Foods: All rows, column "Food" (120 foods)
    - $c_i$: column "Calories"
    - $p_i$: column "Protein(g)"
    - $f_i$: column "Fat(g)"
    - $v_i$: column "VitaminC(mg)"
    - $cost_i$: column "Cost"
    - $x_i$: servings of food $i$ (decision variable, continuous $\geq 0$)

**Nutritional requirements:**
- Calories: $\geq 2000$ kcal
- Protein: $\geq 50$ g
- Vitamin C: $\geq 60$ mg
- Fat: $\leq 70$ g

**Portions may be fractional.** All foods in the file are available for selection. No additional constraints.

---

**Summary:**  
Minimize total cost of servings chosen from all foods in cost.csv, such that the meal plan meets or exceeds 2000 kcal, 50 g protein, 60 mg vitamin C, and does not exceed 70 g fat, with nonnegative (possibly fractional) servings for each food. All parameter values are mapped directly from the specified columns in cost.csv.
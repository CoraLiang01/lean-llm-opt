### Optimization Model

**Sets:**  
Let $\mathcal{F}$ be the set of all foods in cost.csv (see Data Mapping for exact list).

**Parameters (from cost.csv, table_id: file_0_view_0):**  
For each food $f \in \mathcal{F}$:
- $c_f$: Calories per serving (column "Calories")
- $p_f$: Protein per serving (g) (column "Protein(g)")
- $a_f$: Fat per serving (g) (column "Fat(g)")
- $v_f$: VitaminC per serving (mg) (column "VitaminC(mg)")
- $q_f$: Cost per serving (USD) (column "Cost")

**Decision Variables:**  
For each $f \in \mathcal{F}$:
- $x_f \geq 0$: Number of servings of food $f$ (continuous, may be fractional)

**Objective:**  
Minimize total cost:
$$
\min \sum_{f \in \mathcal{F}} q_f x_f
$$

**Constraints:**
- Calorie requirement:
$$
\sum_{f \in \mathcal{F}} c_f x_f \geq 2000
$$
- Protein requirement:
$$
\sum_{f \in \mathcal{F}} p_f x_f \geq 50
$$
- Vitamin C requirement:
$$
\sum_{f \in \mathcal{F}} v_f x_f \geq 60
$$
- Fat limit:
$$
\sum_{f \in \mathcal{F}} a_f x_f \leq 70
$$
- Non-negativity:
$$
x_f \geq 0 \quad \forall f \in \mathcal{F}
$$

---

**Data Mapping:**  
- Foods: $\mathcal{F}$ = all rows in cost.csv, table_id: file_0_view_0, column "Food"
- $c_f$ = column "Calories"
- $p_f$ = column "Protein(g)"
- $a_f$ = column "Fat(g)"
- $v_f$ = column "VitaminC(mg)"
- $q_f$ = column "Cost"
- All constraints and the objective use these parameters as described above.
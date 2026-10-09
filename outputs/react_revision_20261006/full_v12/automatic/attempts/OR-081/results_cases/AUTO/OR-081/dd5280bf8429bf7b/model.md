## Mathematical Model

**Sets:**  
Let $\mathcal{F}$ be the set of all foods in cost.csv (see Data Mapping for exact list).

**Parameters (from cost.csv, table_id: file_0_view_0):**  
For each food $i \in \mathcal{F}$:
- $c_i$: Calories per serving (column "Calories")
- $p_i$: Protein per serving (g) (column "Protein(g)")
- $f_i$: Fat per serving (g) (column "Fat(g)")
- $v_i$: Vitamin C per serving (mg) (column "VitaminC(mg)")
- $cost_i$: Cost per serving (USD) (column "Cost")

**Decision Variables:**  
For each food $i \in \mathcal{F}$:
- $x_i \geq 0$: Number of servings of food $i$ (continuous, may be fractional)

**Objective:**  
Minimize total cost:
$$
\min \sum_{i \in \mathcal{F}} cost_i \, x_i
$$

**Constraints:**
- Calorie requirement:
$$
\sum_{i \in \mathcal{F}} c_i \, x_i \geq 2000
$$

- Protein requirement:
$$
\sum_{i \in \mathcal{F}} p_i \, x_i \geq 50
$$

- Vitamin C requirement:
$$
\sum_{i \in \mathcal{F}} v_i \, x_i \geq 60
$$

- Fat limit:
$$
\sum_{i \in \mathcal{F}} f_i \, x_i \leq 70
$$

- Non-negativity:
$$
x_i \geq 0 \quad \forall i \in \mathcal{F}
$$

---

**Data Mapping:**  
- $\mathcal{F}$: All foods in cost.csv, table_id: file_0_view_0, column "Food"
- $c_i$: "Calories" column, table_id: file_0_view_0, for food $i$
- $p_i$: "Protein(g)" column, table_id: file_0_view_0, for food $i$
- $f_i$: "Fat(g)" column, table_id: file_0_view_0, for food $i$
- $v_i$: "VitaminC(mg)" column, table_id: file_0_view_0, for food $i$
- $cost_i$: "Cost" column, table_id: file_0_view_0, for food $i$
- $x_i$: servings of food $i$ (decision variable)

**All constraints and variables are defined over the full set of foods in the provided data.**
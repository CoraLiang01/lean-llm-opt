## Mathematical Model

**Sets**  
Let $\mathcal{F}$ be the set of all foods in the table (Food column in cost.csv, table_id: file_0_view_0).

**Parameters**  
For each food $i \in \mathcal{F}$ (from cost.csv, table_id: file_0_view_0):
- $c_i$: Calories per serving (Calories)
- $p_i$: Protein per serving (Protein(g))
- $f_i$: Fat per serving (Fat(g))
- $v_i$: Vitamin C per serving (VitaminC(mg))
- $cost_i$: Cost per serving (Cost)

**Decision Variables**  
For each $i \in \mathcal{F}$:
- $x_i \geq 0$: Number of servings of food $i$ (continuous, may be fractional)

**Objective**  
Minimize total cost:
$$
\min \sum_{i \in \mathcal{F}} cost_i \, x_i
$$

**Constraints**
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

**Data Mapping**

- $\mathcal{F}$: All rows in cost.csv, column "Food", table_id: file_0_view_0
- $c_i$: "Calories" column, table_id: file_0_view_0
- $p_i$: "Protein(g)" column, table_id: file_0_view_0
- $f_i$: "Fat(g)" column, table_id: file_0_view_0
- $v_i$: "VitaminC(mg)" column, table_id: file_0_view_0
- $cost_i$: "Cost" column, table_id: file_0_view_0
- $x_i$: servings of food $i$ (decision variable for each $i \in \mathcal{F}$)

All foods in the table are eligible; all constraints and parameters are mapped directly from the specified columns and units. Portions may be fractional (continuous $x_i \geq 0$).
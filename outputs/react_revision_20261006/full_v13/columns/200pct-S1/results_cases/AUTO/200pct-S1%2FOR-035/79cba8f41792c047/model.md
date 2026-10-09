## Mathematical Model

**Sets**  
Let $\mathcal{F}$ be the set of all foods in cost.csv (Food column).

**Parameters**  
For each food $i \in \mathcal{F}$ (from table_id: file_0_view_0, columns as indicated):
- $c_i$ = Calories per serving (Calories)
- $p_i$ = Protein per serving (Protein(g))
- $f_i$ = Fat per serving (Fat(g))
- $v_i$ = Vitamin C per serving (VitaminC(mg))
- $q_i$ = Cost per serving (Cost)

**Decision Variables**  
For each $i \in \mathcal{F}$:
- $x_i \geq 0$: Number of servings of food $i$ (continuous, may be fractional)

**Objective**  
Minimize total cost:
$$
\min \sum_{i \in \mathcal{F}} q_i x_i
$$

**Subject to:**
- Calorie requirement:
$$
\sum_{i \in \mathcal{F}} c_i x_i \geq 2000
$$
- Protein requirement:
$$
\sum_{i \in \mathcal{F}} p_i x_i \geq 50
$$
- Vitamin C requirement:
$$
\sum_{i \in \mathcal{F}} v_i x_i \geq 60
$$
- Fat limit:
$$
\sum_{i \in \mathcal{F}} f_i x_i \leq 70
$$
- Non-negativity:
$$
x_i \geq 0 \quad \forall i \in \mathcal{F}
$$

---

**Data Mapping**

- $\mathcal{F}$: All rows in file_0_view_0, column Food
- $c_i$: file_0_view_0, column Calories
- $p_i$: file_0_view_0, column Protein(g)
- $f_i$: file_0_view_0, column Fat(g)
- $v_i$: file_0_view_0, column VitaminC(mg)
- $q_i$: file_0_view_0, column Cost
- $x_i$: servings of food $i$ (decision variable, continuous $\geq 0$)

**All constraints and parameters are directly mapped from the current cost.csv as described above.**
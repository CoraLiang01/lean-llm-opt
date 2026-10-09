### Mathematical Model

**Sets**  
Let $\mathcal{F}$ be the set of all foods in cost.csv (Food column, table_id: file_0_view_0).

**Parameters**  
For each $i \in \mathcal{F}$ (from table_id: file_0_view_0):
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
\min \sum_{i \in \mathcal{F}} cost_i\, x_i
$$

**Constraints**
1. Calorie requirement:
$$
\sum_{i \in \mathcal{F}} c_i\, x_i \geq 2000
$$

2. Protein requirement:
$$
\sum_{i \in \mathcal{F}} p_i\, x_i \geq 50
$$

3. Vitamin C requirement:
$$
\sum_{i \in \mathcal{F}} v_i\, x_i \geq 60
$$

4. Fat upper bound:
$$
\sum_{i \in \mathcal{F}} f_i\, x_i \leq 70
$$

5. Non-negativity:
$$
x_i \geq 0 \quad \forall i \in \mathcal{F}
$$

---

**Data Mapping**

- $\mathcal{F}$: All rows in cost.csv, column Food, table_id: file_0_view_0
- $c_i$: Calories, table_id: file_0_view_0, column Calories
- $p_i$: Protein(g), table_id: file_0_view_0, column Protein(g)
- $f_i$: Fat(g), table_id: file_0_view_0, column Fat(g)
- $v_i$: VitaminC(mg), table_id: file_0_view_0, column VitaminC(mg)
- $cost_i$: Cost, table_id: file_0_view_0, column Cost
- $x_i$: servings of food $i$ (continuous, $\geq 0$)

**Nutritional requirements:**
- Calories $\geq$ 2000 kcal
- Protein $\geq$ 50 g
- Vitamin C $\geq$ 60 mg
- Fat $\leq$ 70 g

**Portions may be fractional.**
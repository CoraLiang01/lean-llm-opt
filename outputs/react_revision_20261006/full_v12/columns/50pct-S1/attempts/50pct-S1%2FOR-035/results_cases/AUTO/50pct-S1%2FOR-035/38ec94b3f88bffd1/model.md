### Optimization Model

**Sets:**  
Let $\mathcal{F}$ be the set of all foods in cost.csv (see Data Mapping for exact list).

**Parameters (from cost.csv, for each $i \in \mathcal{F}$):**  
$c_i$ = Calories per serving of food $i$  
$p_i$ = Protein (g) per serving of food $i$  
$f_i$ = Fat (g) per serving of food $i$  
$v_i$ = VitaminC (mg) per serving of food $i$  
$cost_i$ = Cost (USD) per serving of food $i$

**Decision Variables:**  
$x_i \geq 0$ = number of servings of food $i$ to include in the meal plan (continuous, may be fractional)

**Objective:**  
Minimize total cost:
$$
\min \sum_{i \in \mathcal{F}} cost_i \, x_i
$$

**Subject to:**
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

- Fat upper bound:
$$
\sum_{i \in \mathcal{F}} f_i \, x_i \leq 70
$$

- Nonnegativity:
$$
x_i \geq 0 \quad \forall i \in \mathcal{F}
$$

---

**Data Mapping:**  
- $\mathcal{F}$: All foods in column "Food" of table_id file_0_view_0 (cost.csv)
- $c_i$: "Calories" column, file_0_view_0, for food $i$
- $p_i$: "Protein(g)" column, file_0_view_0, for food $i$
- $f_i$: "Fat(g)" column, file_0_view_0, for food $i$
- $v_i$: "VitaminC(mg)" column, file_0_view_0, for food $i$
- $cost_i$: "Cost" column, file_0_view_0, for food $i$
- $x_i$: servings of food $i$ (continuous, $\geq 0$)

**All foods and parameters are taken exactly as listed in cost.csv (table_id file_0_view_0).**
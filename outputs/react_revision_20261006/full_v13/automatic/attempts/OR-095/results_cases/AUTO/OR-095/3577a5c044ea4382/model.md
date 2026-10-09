## Symbolic Mathematical Model

### Sets
- $I$: Set of widgets, $I = \{\text{Widget1}, \ldots, \text{Widget141}\}$ (from file_1_view_0, column Product)

### Parameters (Data Mapping)
- $a_i$: Labor hours required per unit of widget $i$ (file_1_view_0, LaborHours, $i \in I$)
- $b_i$: Material A required per unit of widget $i$ (file_1_view_0, MaterialA, $i \in I$)
- $c_i$: Material B required per unit of widget $i$ (file_1_view_0, MaterialB, $i \in I$)
- $p_i$: Base profit per unit of widget $i$ (file_1_view_0, Profit, $i \in I$)
- $L^{\max}$: Monthly labor hour limit = 5000 (file_0_view_0, Resource = LaborHours, MonthlyLimit)
- $A^{\max}$: Monthly Material A limit = 24000 (file_0_view_0, Resource = MaterialA, MonthlyLimit)
- $B^{\max}$: Monthly Material B limit = 15000 (file_0_view_0, Resource = MaterialB, MonthlyLimit)
- $r_X$: CatalystX produced per unit of Widget3 = 5 kg/unit (problem statement)
- $P_X^{\text{sell}}$: Sale price of CatalystX = 300 $/kg$ (problem statement)
- $P_X^{\text{dispose}}$: Disposal cost of CatalystX = 200 $/kg$ (problem statement)
- $S_X^{\max}$: Maximum CatalystX sales per month = 1500 kg (problem statement)

### Decision Variables
- $x_i \geq 0$: Number of units of widget $i$ produced, $\forall i \in I$ (continuous)
- $y_X \geq 0$: Amount (kg) of CatalystX sold (continuous)
- $z_X \geq 0$: Amount (kg) of CatalystX disposed (continuous)

### Objective
Maximize total profit:
$$
\max \left\{
\sum_{i \in I} p_i x_i
+ P_X^{\text{sell}} y_X
- P_X^{\text{dispose}} z_X
\right\}
$$

### Constraints

1. **Labor hours limit**
$$
\sum_{i \in I} a_i x_i \leq L^{\max}
$$

2. **Material A limit**
$$
\sum_{i \in I} b_i x_i \leq A^{\max}
$$

3. **Material B limit**
$$
\sum_{i \in I} c_i x_i \leq B^{\max}
$$

4. **CatalystX production and balance**
$$
r_X x_{\text{Widget3}} = y_X + z_X
$$

5. **CatalystX sales cap**
$$
0 \leq y_X \leq S_X^{\max}
$$

6. **Nonnegativity**
$$
x_i \geq 0 \quad \forall i \in I
$$
$$
y_X \geq 0
$$
$$
z_X \geq 0
$$

---

### Data Mapping

- $I$: file_1_view_0, column Product
- $a_i$: file_1_view_0, LaborHours, row with Product $i$
- $b_i$: file_1_view_0, MaterialA, row with Product $i$
- $c_i$: file_1_view_0, MaterialB, row with Product $i$
- $p_i$: file_1_view_0, Profit, row with Product $i$
- $L^{\max}$: file_0_view_0, Resource = LaborHours, MonthlyLimit
- $A^{\max}$: file_0_view_0, Resource = MaterialA, MonthlyLimit
- $B^{\max}$: file_0_view_0, Resource = MaterialB, MonthlyLimit
- $r_X$: 5 (problem statement, Widget3 only)
- $P_X^{\text{sell}}$: 300 (problem statement)
- $P_X^{\text{dispose}}$: 200 (problem statement)
- $S_X^{\max}$: 1500 (problem statement)
- $x_i$: production quantity of widget $i$
- $y_X$: CatalystX sold
- $z_X$: CatalystX disposed

---

All sets, parameters, variables, objective, and constraints are mapped directly to the provided data and problem statement.
## Abstract Mathematical Model

Let:
- $I$ = set of widgets, indexed by $i$ (from all Product in product_resources.csv)
- $x_i$ = number of units of widget $i$ to produce (integer, $x_i \geq 0$)
- $y$ = amount (kg) of CatalystX sold (continuous, $y \geq 0$)

Parameters:
- $a_i$ = labor hours per unit of widget $i$ (LaborHours, file_1_view_0)
- $b_i$ = MaterialA per unit of widget $i$ (MaterialA, file_1_view_0)
- $c_i$ = MaterialB per unit of widget $i$ (MaterialB, file_1_view_0)
- $p_i$ = base profit per unit of widget $i$ (Profit, file_1_view_0)
- $L$ = monthly labor hour limit (MonthlyLimit for Resource = LaborHours, file_0_view_0)
- $M_A$ = monthly MaterialA limit (MonthlyLimit for Resource = MaterialA, file_0_view_0)
- $M_B$ = monthly MaterialB limit (MonthlyLimit for Resource = MaterialB, file_0_view_0)
- $r$ = CatalystX generated per unit of Widget3 ($r = 5$ kg/unit)
- $q$ = maximum CatalystX sales per month ($q = 1500$ kg)
- $s$ = sale price per kg CatalystX ($s = 300$)
- $d$ = disposal cost per kg unsold CatalystX ($d = 200$)

Let $i^*$ denote the index of Widget3 in $I$.

### Objective:
\[
\max \left\{ \sum_{i \in I} p_i x_i + s y - d \left[ r x_{i^*} - y \right] \right\}
\]
where $r x_{i^*}$ is total CatalystX produced, $y$ is amount sold, and $r x_{i^*} - y$ is amount disposed.

### Constraints:
1. Labor hours:
   \[
   \sum_{i \in I} a_i x_i \leq L
   \]
2. Material A:
   \[
   \sum_{i \in I} b_i x_i \leq M_A
   \]
3. Material B:
   \[
   \sum_{i \in I} c_i x_i \leq M_B
   \]
4. CatalystX sales cap:
   \[
   0 \leq y \leq \min\{ r x_{i^*},\ q \}
   \]
   (i.e., cannot sell more than produced or more than market cap)
5. Integrality and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]
   \[
   y \geq 0
   \]

### Data Mapping

#### Decision Variables
- $x_i$: integer, number of units to produce for each Product (Product, file_1_view_0)
- $y$: continuous, kg of CatalystX sold

#### Parameters
- $a_i$: LaborHours (file_1_view_0, column LaborHours, key Product)
- $b_i$: MaterialA (file_1_view_0, column MaterialA, key Product)
- $c_i$: MaterialB (file_1_view_0, column MaterialB, key Product)
- $p_i$: Profit (file_1_view_0, column Profit, key Product)
- $L$: MonthlyLimit where Resource = LaborHours (file_0_view_0)
- $M_A$: MonthlyLimit where Resource = MaterialA (file_0_view_0)
- $M_B$: MonthlyLimit where Resource = MaterialB (file_0_view_0)
- $i^*$: Product = Widget3 (file_1_view_0)
- $r = 5$, $q = 1500$, $s = 300$, $d = 200$ (from user description)

#### Source Tables
- file_1_view_0: product_resources.csv (columns: Product, LaborHours, MaterialA, MaterialB, Profit)
- file_0_view_0: resource_limits.csv (columns: Resource, MonthlyLimit)

---

This model maximizes total profit from widget production and CatalystX sales, minus disposal costs, subject to all resource and market constraints.
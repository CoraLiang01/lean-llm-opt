#### Abstract Mathematical Model

Let:
- $I$ = set of components, indexed by $i$ (from unit_price.csv, Unnamed: 0)
- $W$ = set of workshops, indexed by $w$ (from processing_time_unit.csv and total_working_hours.csv, Unnamed: 0/workshop)
- $p_i$ = unit price of component $i$ (from unit_price.csv, unit_price)
- $a_{wi}$ = unit processing time of component $i$ in workshop $w$ (from processing_time_unit.csv, $w$ as row Unnamed: 0, $i$ as column)
- $b_w$ = total available working hours in workshop $w$ (from total_working_hours.csv, total_hours)
- $x_i$ = number of units to produce of component $i$ (decision variable, integer, $x_i \geq 0$)

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Constraints:**
- Workshop capacity constraints:
$$
\sum_{i \in I} a_{wi} x_i \leq b_w, \quad \forall w \in W
$$
- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (components): file_1_view_0, column Unnamed: 0
- $W$ (workshops): file_0_view_0, row Unnamed: 0; file_2_view_0, column workshop
- $p_i$: file_1_view_0, column unit_price, key Unnamed: 0
- $a_{wi}$: file_0_view_0, row Unnamed: 0 ($w$), column $i$ (e.g., C1, C2, ..., C111)
- $b_w$: file_2_view_0, column total_hours, key workshop
- $x_i$: decision variable for each $i \in I$

All sets, parameters, and constraints are indexed and mapped exactly as in the source data.
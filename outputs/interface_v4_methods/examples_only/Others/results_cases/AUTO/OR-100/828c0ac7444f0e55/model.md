#### Abstract Mathematical Model

Let:
- $I$ = set of component types, indexed by $i$ (from unit_price.csv, column Unnamed: 0)
- $W$ = set of workshops, indexed by $w$ (from processing_time_unit.csv and total_working_hours.csv, row/column Unnamed: 0)
- $x_i$ = number of units of component $i$ to produce (decision variable, nonnegative integer)

Parameters:
- $p_i$ = unit price of component $i$ (from unit_price.csv, column unit_price)
- $a_{wi}$ = unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, entry at row $w$, column $i$)
- $c_w$ = total available working hours in workshop $w$ (from total_working_hours.csv, column total_hours)

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to (for all $w \in W$):
$$
\sum_{i \in I} a_{wi} x_i \leq c_w
$$

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (component types): file_1_view_0, column Unnamed: 0
- $W$ (workshops): file_0_view_0, row Unnamed: 0; file_2_view_0, column workshop
- $p_i$: file_1_view_0, column unit_price, key Unnamed: 0
- $a_{wi}$: file_0_view_0, row Unnamed: 0 (workshop), column $i$ (component)
- $c_w$: file_2_view_0, column total_hours, key workshop

All indices, parameters, and constraints are mapped directly to the original file columns and business identifiers.
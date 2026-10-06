## Abstract Mathematical Model

Let:
- $I$ = set of components, indexed by $i$ (from unit_price.csv, column Unnamed: 0)
- $W$ = set of workshops, indexed by $w$ (from processing_time_unit.csv and total_working_hours.csv, row/column Unnamed: 0)
- $p_i$ = unit price of component $i$ (from unit_price.csv, column unit_price)
- $a_{wi}$ = unit processing time of component $i$ in workshop $w$ (from processing_time_unit.csv, entry at row $w$, column $i$)
- $c_w$ = total available working hours in workshop $w$ (from total_working_hours.csv, column total_hours)
- $x_i$ = number of units of component $i$ to produce (decision variable, nonnegative integer)

### Objective
$$
\max \sum_{i \in I} p_i \, x_i
$$

### Constraints

#### 1. Workshop Capacity Constraints
$$
\sum_{i \in I} a_{wi} \, x_i \leq c_w, \quad \forall w \in W
$$

#### 2. Nonnegativity and Integrality
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

---

## Data Mapping

- $I$ (components): All values in unit_price.csv, column Unnamed: 0 (e.g., C1, C2, ..., C111)
- $W$ (workshops): All values in processing_time_unit.csv, column Unnamed: 0 and total_working_hours.csv, column workshop (e.g., Casting, Milling, Finishing, Assembly, QA & Packaging)
- $p_i$: unit_price.csv, columns Unnamed: 0 (component ID), unit_price
- $a_{wi}$: processing_time_unit.csv, row Unnamed: 0 = $w$, column $i$ (C1...C111)
- $c_w$: total_working_hours.csv, columns workshop, total_hours

Each $x_i$ is the number of units of component $i$ to produce.

---

## Summary

- Maximize total output value.
- Each workshop's total processing time across all components cannot exceed its available working hours.
- All variables are nonnegative integers.
- All parameters and indices are mapped directly from the supplied CSV files and columns.
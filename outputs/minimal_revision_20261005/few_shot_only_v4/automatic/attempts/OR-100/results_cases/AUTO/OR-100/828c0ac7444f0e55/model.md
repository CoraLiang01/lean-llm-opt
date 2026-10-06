**Abstract Mathematical Model**

**Index Sets**
- $I$: set of components (from unit_price.csv, column Unnamed: 0; e.g., $i \in I$)
- $W$: set of workshops (from processing_time_unit.csv and total_working_hours.csv, row/column Unnamed: 0; e.g., $w \in W$)

**Parameters**
- $p_i$: unit price of component $i$ (from unit_price.csv, column unit_price)
- $a_{wi}$: unit processing time of component $i$ in workshop $w$ (from processing_time_unit.csv, row $w$, column $i$)
- $b_w$: total available working hours in workshop $w$ (from total_working_hours.csv, column total_hours)

**Decision Variables**
- $x_i$: number of units to produce of component $i$; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints**
\[
\sum_{i \in I} a_{wi} x_i \leq b_w, \quad \forall w \in W
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All values in unit_price.csv, column Unnamed: 0 (component IDs, e.g., C1, C2, ..., C111)
- $W$: All values in processing_time_unit.csv, column Unnamed: 0 and total_working_hours.csv, column workshop (e.g., Casting, Milling, Finishing, Assembly, QA & Packaging)
- $p_i$: unit_price.csv, columns Unnamed: 0 (ID), unit_price (value)
- $a_{wi}$: processing_time_unit.csv, row Unnamed: 0 = $w$, column $i$ (C1, ..., C111)
- $b_w$: total_working_hours.csv, columns workshop (ID), total_hours (value)

**Notes**
- Each $x_i$ is a nonnegative integer (whole units produced).
- Each workshop's total time used by all components cannot exceed its available hours.
- All 111 components and all 5 workshops are included, with their exact IDs and mappings as above.
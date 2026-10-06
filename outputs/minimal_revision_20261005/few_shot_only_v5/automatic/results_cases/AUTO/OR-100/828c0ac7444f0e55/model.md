**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of components (from processing_time_unit.csv columns $C1, C2, ..., C111$)
- $W$: set of workshops (from processing_time_unit.csv rows and total_working_hours.csv "workshop")

**Parameters:**
- $p_i$: unit price of component $i \in I$ (from unit_price.csv, column "unit_price", key "Unnamed: 0")
- $a_{wi}$: unit processing time of component $i$ in workshop $w$ (from processing_time_unit.csv, row $w$, column $i$)
- $b_w$: total available working hours in workshop $w$ (from total_working_hours.csv, column "total_hours", key "workshop")

**Decision Variables:**
- $x_i$: number of units to produce of component $i \in I$ ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_{wi} x_i \leq b_w, \quad \forall w \in W
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All columns except "Unnamed: 0" in /.../processing_time_unit.csv
- $W$: All rows in /.../processing_time_unit.csv, and "workshop" in /.../total_working_hours.csv
- $p_i$: /.../unit_price.csv, column "unit_price", key "Unnamed: 0" = $i$
- $a_{wi}$: /.../processing_time_unit.csv, row "Unnamed: 0" = $w$, column $i$
- $b_w$: /.../total_working_hours.csv, column "total_hours", key "workshop" = $w$
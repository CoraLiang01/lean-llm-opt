##### Mathematical Model

Let:
- $I$ = set of component types, indexed by $i$ (from all Unnamed: 0 in unit_price.csv and columns in processing_time_unit.csv except "Unnamed: 0")
- $W$ = set of workshops, indexed by $w$ (from all workshop in total_working_hours.csv and rows in processing_time_unit.csv)
- $x_i$ = number of units of component $i$ to produce (decision variable, $x_i \geq 0$, integer)
- $p_i$ = unit price of component $i$
- $a_{w,i}$ = unit processing time of component $i$ in workshop $w$
- $c_w$ = total available working hours in workshop $w$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_{w,i} x_i \leq c_w, \quad \forall w \in W
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

##### Data Mapping

- $I$: All component types from file_1_view_0.Unnamed: 0 and file_0_view_0 columns (excluding "Unnamed: 0")
- $W$: All workshops from file_2_view_0.workshop and file_0_view_0.Unnamed: 0
- $p_i$: file_1_view_0.unit_price, keyed by file_1_view_0.Unnamed: 0 = $i$
- $a_{w,i}$: file_0_view_0, row Unnamed: 0 = $w$, column $i$
- $c_w$: file_2_view_0.total_hours, keyed by file_2_view_0.workshop = $w$
- $x_i$: decision variable, integer, for each $i \in I$
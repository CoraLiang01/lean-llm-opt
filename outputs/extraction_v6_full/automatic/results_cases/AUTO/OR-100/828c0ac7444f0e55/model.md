#### Abstract Mathematical Model

Let:
- $I$ = set of component types (from C1 to C111, as in unit_price.csv and processing_time_unit.csv columns)
- $W$ = set of workshops (Casting, Milling, Finishing, Assembly, QA & Packaging, as in processing_time_unit.csv rows and total_working_hours.csv)
- $x_i$ = number of units of component $i \in I$ to produce (decision variable, integer, $x_i \geq 0$)
- $p_i$ = unit price of component $i$ (from unit_price.csv)
- $a_{wi}$ = unit processing time of component $i$ in workshop $w$ (from processing_time_unit.csv)
- $c_w$ = total available working hours in workshop $w$ (from total_working_hours.csv)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_{wi} x_i \leq c_w \qquad \forall w \in W
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

#### Data Mapping

- $I$ (component types): file_1_view_0.Unnamed: 0 (unit_price.csv), file_0_view_0.C1 ... C111 (processing_time_unit.csv columns)
- $W$ (workshops): file_0_view_0.Unnamed: 0 (processing_time_unit.csv rows), file_2_view_0.workshop (total_working_hours.csv)
- $p_i$: file_1_view_0.unit_price, indexed by file_1_view_0.Unnamed: 0
- $a_{wi}$: file_0_view_0, value at [row: Unnamed: 0 = $w$, column: $i$]
- $c_w$: file_2_view_0.total_hours, indexed by file_2_view_0.workshop

---

**All data, indices, and parameters are mapped directly from the supplied CSV files and columns as described above.**
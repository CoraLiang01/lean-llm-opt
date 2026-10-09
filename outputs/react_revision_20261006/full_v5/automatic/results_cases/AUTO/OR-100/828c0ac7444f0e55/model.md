##### Mathematical Model

Let:
- $I$ = set of component types (from file_1_view_0.Unnamed: 0, e.g., $C1, C2, ..., C111$)
- $R$ = set of workshops (from file_0_view_0.Unnamed: 0 and file_2_view_0.workshop, e.g., Casting, Milling, Finishing, Assembly, QA & Packaging)
- $x_i$ = number of units to produce of component $i \in I$ (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)
- $p_i$ = unit price of component $i$ (from file_1_view_0.unit_price)
- $a_{ri}$ = unit processing time of component $i$ in workshop $r$ (from file_0_view_0, row $r$, column $i$)
- $c_r$ = total available working hours in workshop $r$ (from file_2_view_0.total_hours)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_{ri} x_i \leq c_r \qquad \forall r \in R
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

##### Data Mapping

- $I$: file_1_view_0.Unnamed: 0 (component IDs, e.g., C1, ..., C111)
- $R$: file_0_view_0.Unnamed: 0 and file_2_view_0.workshop (workshop names, e.g., Casting, Milling, Finishing, Assembly, QA & Packaging)
- $p_i$: file_1_view_0.unit_price, keyed by file_1_view_0.Unnamed: 0
- $a_{ri}$: file_0_view_0, row indexed by Unnamed: 0 (workshop), column indexed by component ID (C1, ..., C111)
- $c_r$: file_2_view_0.total_hours, keyed by file_2_view_0.workshop
- $x_i$: decision variable, for each $i \in I$ (component ID)

All index sets, parameters, and constraints are mapped directly from the supplied files and columns.
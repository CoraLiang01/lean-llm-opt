#### Mathematical Model

Let:
- $I$ = set of component types (indexed by $i$), as given by file_1_view_0.Unnamed: 0 and file_0_view_0 columns $C1,\ldots,C111$
- $K$ = set of workshops (indexed by $k$), as given by file_0_view_0.Unnamed: 0 and file_2_view_0.workshop
- $x_i$ = number of units of component $i$ to produce (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)

Parameters:
- $p_i$ = unit price of component $i$ (from file_1_view_0.unit_price)
- $a_{ki}$ = unit processing time of component $i$ in workshop $k$ (from file_0_view_0, row $k$, column $i$)
- $b_k$ = total available working hours in workshop $k$ (from file_2_view_0.total_hours)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to (for all $k \in K$):
\[
\sum_{i \in I} a_{ki} x_i \leq b_k
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$ (component types): file_1_view_0.Unnamed: 0 and file_0_view_0 columns $C1,\ldots,C111$
- $K$ (workshops): file_0_view_0.Unnamed: 0 and file_2_view_0.workshop
- $p_i$: file_1_view_0.unit_price, indexed by file_1_view_0.Unnamed: 0
- $a_{ki}$: file_0_view_0, row indexed by Unnamed: 0 (workshop), column indexed by $C1,\ldots,C111$ (component)
- $b_k$: file_2_view_0.total_hours, indexed by file_2_view_0.workshop
- $x_i$: decision variable, nonnegative integer, for each $i \in I$
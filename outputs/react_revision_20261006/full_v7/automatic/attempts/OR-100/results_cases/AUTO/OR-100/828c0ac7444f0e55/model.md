##### Mathematical Model

Let:
- $I$ = set of component types, indexed by $i$ (from all Unnamed: 0 in file_1_view_0 and columns in file_0_view_0)
- $J$ = set of workshops, indexed by $j$ (from all Unnamed: 0 in file_0_view_0 and workshop in file_2_view_0)
- $x_i$ = number of units of component $i$ to produce (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)
- $p_i$ = unit price of component $i$ (from unit_price in file_1_view_0)
- $a_{ji}$ = unit processing time of component $i$ in workshop $j$ (from file_0_view_0, row $j$, column $i$)
- $c_j$ = total available working hours in workshop $j$ (from total_hours in file_2_view_0)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to (for all $j \in J$):
\[
\sum_{i \in I} a_{ji} x_i \leq c_j
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: All component types from file_1_view_0.Unnamed: 0 and file_0_view_0 columns C1...C111
- $J$: All workshops from file_0_view_0.Unnamed: 0 and file_2_view_0.workshop
- $p_i$: file_1_view_0.unit_price, keyed by Unnamed: 0
- $a_{ji}$: file_0_view_0, row Unnamed: 0 = $j$, column $i$
- $c_j$: file_2_view_0.total_hours, keyed by workshop

Decision variables:
- $x_i$: number of units of component $i$ to produce, integer, $\geq 0$
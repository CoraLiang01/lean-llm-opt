ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of components, indexed by $i$ (from file_1_view_0.Unnamed: 0 and file_0_view_0.C1...C111)
- $K$: set of workshops, indexed by $k$ (from file_2_view_0.workshop and file_0_view_0.Unnamed: 0)

Parameters:
- $p_i$: unit price of component $i$ (from file_1_view_0.unit_price, key Unnamed: 0)
- $a_{ki}$: unit processing time of component $i$ in workshop $k$ (from file_0_view_0, row Unnamed: 0 = $k$, column $i$)
- $b_k$: total available working hours in workshop $k$ (from file_2_view_0.total_hours, key workshop)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of component $i$ to produce

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} a_{ki} x_i \leq b_k \qquad \forall k \in K
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping:
- $I$ (components): file_1_view_0.Unnamed: 0, file_0_view_0.C1...C111
- $K$ (workshops): file_2_view_0.workshop, file_0_view_0.Unnamed: 0
- $p_i$: file_1_view_0.unit_price, key Unnamed: 0
- $a_{ki}$: file_0_view_0, row Unnamed: 0 = $k$, column $i$
- $b_k$: file_2_view_0.total_hours, key workshop

Variable:
- $x_i$: number of units of component $i$ to produce (nonnegative integer)

Objective:
- Maximize total output value

Constraints:
- For each workshop $k$, total processing time used by all components does not exceed available hours $b_k$
- All $x_i$ are nonnegative integers
Mathematical Model

Sets:
- $I$: set of components, indexed by $i$ (component IDs from file_1_view_0["Unnamed: 0"])
- $K$: set of workshops, indexed by $k$ (workshop names from file_0_view_0["Unnamed: 0"] and file_2_view_0["workshop"])

Parameters:
- $p_i$: unit price of component $i$ (file_1_view_0["unit_price"])
- $a_{ki}$: unit processing time of component $i$ in workshop $k$ (file_0_view_0, row $k$, column $i$)
- $b_k$: total available working hours in workshop $k$ (file_2_view_0["total_hours"])

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

Data Mapping

- $I$: file_1_view_0["Unnamed: 0"]
- $K$: file_0_view_0["Unnamed: 0"] and file_2_view_0["workshop"]
- $p_i$: file_1_view_0["unit_price"], matched by component ID
- $a_{ki}$: file_0_view_0, row with Unnamed: 0 = $k$, column $i$
- $b_k$: file_2_view_0["total_hours"], matched by workshop name $k$
- $x_i$: production quantity for component $i$ (decision variable)
ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of vehicle types (indexed by $i$), from file_1_view_0 ProductName.

Parameters:
- $p_i$: profit per unit of vehicle $i$ (file_1_view_0, Value, for ProductName $i$)
- $w_i$: weight (space requirement) per unit of vehicle $i$ (file_1_view_0, Weight, for ProductName $i$)
- $C$: total inventory capacity (file_0_view_0, Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

DATA MAPPING

- $I$: All ProductName values in file_1_view_0
- $p_i$: file_1_view_0, column Value, for ProductName $i$
- $w_i$: file_1_view_0, column Weight, for ProductName $i$
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for ProductName $i$
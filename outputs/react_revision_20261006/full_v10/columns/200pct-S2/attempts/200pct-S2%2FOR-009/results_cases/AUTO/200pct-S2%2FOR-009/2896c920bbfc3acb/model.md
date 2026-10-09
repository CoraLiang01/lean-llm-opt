ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $I$ be the set of areas, with each area identified by file_1_view_0.ProductName.

Parameters:
- $v_i$: development benefit per unit in area $i$ (file_1_view_0.Value)
- $w_i$: development resource required per unit in area $i$ (file_1_view_0.Weight)
- $C$: overall development capacity (file_0_view_0.Capacity)

Decision Variables:
- $x_i$: scale of development per day in area $i$, $x_i \geq 0$ and integer, for all $i \in I$

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

DATA MAPPING

- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C$: file_0_view_0.Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, indexed by file_1_view_0.ProductName)
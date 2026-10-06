#### Abstract Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names from file_1_view_0.ProductName
- $x_i$ = scale of development per day in area $i$ (decision variable), $x_i \geq 0$, integer

Parameters:
- $b_i$ = development benefit of area $i$ (from file_1_view_0.Value)
- $w_i$ = development weight (resource consumption) of area $i$ (from file_1_view_0.Weight)
- $C$ = overall development capacity (from file_0_view_0.Capacity)

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $b_i$: file_1_view_0, column Value, indexed by ProductName
- $w_i$: file_1_view_0, column Weight, indexed by ProductName
- $C$: file_0_view_0, column Capacity (single value)
- $x_i$: decision variable for each ProductName in file_1_view_0

All data is used in original file and row order, with explicit business identifiers.
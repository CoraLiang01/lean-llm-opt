Mathematical Model

Sets:
Let $I$ be the set of production plants (indexed by $i$), with $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$.
Let $J$ be the set of retail outlets (indexed by $j$), with $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$.

Parameters:
$d_j$: daily demand at outlet $j \in J$ (from file_0_view_0, column demand).
$s_i$: daily production capacity at plant $i \in I$ (from file_1_view_0, column supply_capacity).
$c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4).

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous).

Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction at each outlet:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Plant capacity at each plant:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

Sets:
- $I$ (plants): supplier_id in file_1_view_0 and file_2_view_0, in source order: S1, S2, S3, S4
- $J$ (outlets): customer_id in file_0_view_0 and columns transportation_cost_to_C1, ..., transportation_cost_to_C4 in file_2_view_0, in source order: C1, C2, C3, C4

Parameters:
- $d_j$: file_0_view_0, column demand, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (e.g., transportation_cost_to_C1 for $j$ = C1)

Variables:
- $x_{ij}$: for all $i \in I$, $j \in J$ (continuous, $\geq 0$)

Constraints:
- Demand: for each $j$ in file_0_view_0 customer_id, sum over $i$ in file_1_view_0 supplier_id
- Supply: for each $i$ in file_1_view_0 supplier_id, sum over $j$ in file_0_view_0 customer_id

Objective:
- Minimize $\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$ using $c_{ij}$ from file_2_view_0 as above

No data is omitted or aggregated; all identifiers and coefficients are preserved as in the current Observation.
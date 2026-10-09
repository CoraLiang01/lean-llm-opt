#### Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$), as given in the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from file_2_view_0, columns $J$, rows $I$).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
- Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
- Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

#### Data Mapping

- $I$ (distribution centers): all "Unnamed: 0" values in file_1_view_0 (supply_capacity.csv), in source order.
- $J$ (customer groups): all "customer" values in file_0_view_0 (customer_demand.csv), in source order.
- $d_j$: file_0_view_0, column "demand", for each $j$.
- $s_i$: file_1_view_0, column "supply_capacity", for each $i$.
- $c_{ij}$: file_2_view_0, entry at row $i$ ("Unnamed: 0"), column $j$ (customer group), for all $i \in I$, $j \in J$.
- $x_{ij}$: decision variable for each $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are defined exactly as in the current source data, preserving source order and identifiers.
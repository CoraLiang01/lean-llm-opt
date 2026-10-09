Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers), indexed by $i$.
Let $J$ be the set of customer groups, indexed by $j$.

Parameters:
$d_j$: demand of customer group $j$ (from file_0_view_0, column "demand", key "customer_id").
$s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity", key "supplier_id").
$c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_{j}", key "supplier_id").

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

Sets:
- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.

Parameters:
- $d_j$: file_0_view_0, column "demand", key "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", key "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}" (column_id_mapping in Observation).

Variables:
- $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

Constraints:
- Demand: for each $j$ in $J$, sum over $i$ in $I$.
- Supply: for each $i$ in $I$, sum over $j$ in $J$.
- Non-negativity: for all $i$ in $I$, $j$ in $J$.

All indices, parameters, and coefficients are mapped exactly as above from the current source data.
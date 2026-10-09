#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the identifiers in the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand_units", indexed by "customer_id").
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity_units", indexed by "supplier_id").
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_{j}", row "supplier_id" = $i$).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand_units", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity_units", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}" (where $j$ is the customer_id).

- $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and source order. No data is omitted or aggregated.
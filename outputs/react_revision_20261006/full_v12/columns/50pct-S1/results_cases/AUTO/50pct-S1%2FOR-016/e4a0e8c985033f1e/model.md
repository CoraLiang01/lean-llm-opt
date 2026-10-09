#### Mathematical Model

Let $I$ be the set of distribution centers (supplier_id from file_1), $J$ the set of customer groups (customer_id from file_0).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from $i$ to $j$.

Subject to:
- Demand satisfaction:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
where $d_j$ is the demand of customer $j$.
- Supply capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
where $s_i$ is the supply capacity of distribution center $i$.
- Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source column "supplier_id".
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source column "customer_id".
- $d_j$: file_0_view_0, column "demand_units", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity_units", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0 (transportation_costs.csv), row "supplier_id" = $i$, column "transportation_cost_to_{j}" (e.g., "transportation_cost_to_C1" for $j$ = C1).

Index sets, parameters, and all coefficients are to be taken exactly as listed in the current CSV data, preserving all identifiers and source order.
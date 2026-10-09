#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, columns "transportation_cost_to_Ck").

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", indexed by customer_id.
- $s_i$: file_1_view_0, column "supply_capacity", indexed by supplier_id.
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}$", where $j$ is customer_id.

Index sets, parameters, and cost matrix are defined exactly as in the current source data, preserving all identifiers and source order.
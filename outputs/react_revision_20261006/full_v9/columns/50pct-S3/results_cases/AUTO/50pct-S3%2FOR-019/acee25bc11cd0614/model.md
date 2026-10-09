#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands), as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv, column "demand", table_id: file_0_view_0).
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0).
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv, columns "transportation_cost_to_demand*", table_id: file_2_view_0).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

#### Data Mapping

- $I$ (suppliers): All "supplier_id" in supply_capacity.csv (table_id: file_1_view_0), and as row ids in transportation_costs.csv (table_id: file_2_view_0).
- $J$ (customers): All "customer_id" in customer_demand.csv (table_id: file_0_view_0), and as column suffixes in transportation_costs.csv (table_id: file_2_view_0).
- $d_j$: For each $j \in J$, from customer_demand.csv, column "demand", table_id: file_0_view_0.
- $s_i$: For each $i \in I$, from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0.
- $c_{ij}$: For each $i \in I$, $j \in J$, from transportation_costs.csv, table_id: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}$".

Index sets, parameters, and all coefficients are to be taken exactly as listed in the current source data, preserving all identifiers and source order.
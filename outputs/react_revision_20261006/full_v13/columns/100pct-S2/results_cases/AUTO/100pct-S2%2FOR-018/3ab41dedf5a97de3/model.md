##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j \in J$ (from "customer_demand.csv").
- $s_i$: supply capacity of distribution center $i \in I$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv").

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): all "supplier_id" in "supply_capacity.csv" (table_id: file_1_view_0)
- $J$ (customer groups): all "customer_id" in "customer_demand.csv" (table_id: file_0_view_0)
- $d_j$: "demand" column in "customer_demand.csv" (table_id: file_0_view_0, key: customer_id)
- $s_i$: "supply_capacity" column in "supply_capacity.csv" (table_id: file_1_view_0, key: supplier_id)
- $c_{ij}$: "transportation_costs.csv" (table_id: file_2_view_0), with row key "supplier_id" (distribution center), column key "transportation_cost_to_{customer_id}" (customer group), using the mapping:
  - For each $i \in I$, $j \in J$, $c_{ij}$ is the value in the row with "supplier_id" = $i$ and column "transportation_cost_to_{j}"

Index sets, parameters, and all coefficients are defined exactly as in the current data, preserving all identifiers and source order.
##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j \in J$ (from "customer_demand.csv").
- $s_i$: supply capacity of distribution center $i \in I$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv").

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

- $I$: All supplier_id in "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id)
- $J$: All customer_id in "customer_demand.csv" (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand for customer $j$ from "customer_demand.csv" (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply_capacity for supplier $i$ from "supply_capacity.csv" (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation_costs from supplier $i$ to customer $j$ from "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id})

Index sets, parameters, and cost matrix are defined exactly as in the current data, preserving all identifiers and source order.
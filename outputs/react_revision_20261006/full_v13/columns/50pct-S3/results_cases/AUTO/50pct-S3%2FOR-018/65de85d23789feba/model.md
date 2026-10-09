##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv).
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from transportation_costs.csv).

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

- $I$ (distribution centers): all unique values of supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id)
- $J$ (customer groups): all unique values of customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand for customer group $j$ from customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply capacity for distribution center $i$ from supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ from transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id})

Index sets, parameters, and cost matrix are defined exactly as in the current source data, preserving all identifiers and coefficients. No data is omitted or aggregated.
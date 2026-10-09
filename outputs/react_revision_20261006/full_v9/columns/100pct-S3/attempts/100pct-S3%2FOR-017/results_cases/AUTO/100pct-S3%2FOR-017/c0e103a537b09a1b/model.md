#### Mathematical Model

Let $I$ be the set of suppliers (from the supplier_id column of supply_capacity.csv and transportation_costs.csv), and $J$ be the set of customer groups (from the customer_id column of customer_demand.csv and the transportation_cost_to_C* columns of transportation_costs.csv).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

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

#### Data Mapping

- $I$ (suppliers): All supplier_id values in supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0), in source order.
- $J$ (customers): All customer_id values in customer_demand.csv (file_0_view_0) and all transportation_cost_to_C* columns in transportation_costs.csv (file_2_view_0), in source order.
- $d_j$: For each $j \in J$, demand from customer_demand.csv (file_0_view_0, columns: customer_id, demand).
- $s_i$: For each $i \in I$, supply_capacity from supply_capacity.csv (file_1_view_0, columns: supplier_id, supply_capacity).
- $c_{ij}$: For each $i \in I$, $j \in J$, transportation_costs.csv (file_2_view_0), row: supplier_id $=i$, column: transportation_cost_to_$j$.

Variable:
- $x_{ij}$: Quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving source order and identifiers. No data is omitted or aggregated.
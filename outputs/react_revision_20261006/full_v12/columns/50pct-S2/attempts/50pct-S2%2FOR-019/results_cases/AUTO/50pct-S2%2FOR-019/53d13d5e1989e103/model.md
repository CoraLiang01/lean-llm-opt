#### Abstract Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands), as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv).
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv).

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

- $I$ (suppliers): All unique values in column supplier_id of supply_capacity.csv (file_1_view_0) and row axis of transportation_costs.csv (file_2_view_0).
- $J$ (customers): All unique values in column customer_id of customer_demand.csv (file_0_view_0) and column axis of transportation_costs.csv (file_2_view_0).
- $d_j$: demand for customer $j$ from column demand in customer_demand.csv (file_0_view_0), indexed by customer_id.
- $s_i$: supply_capacity for supplier $i$ from column supply_capacity in supply_capacity.csv (file_1_view_0), indexed by supplier_id.
- $c_{ij}$: transportation_cost_to_demandX for each supplier_id and customer_id pair from transportation_costs.csv (file_2_view_0), with row axis supplier_id and column axis transportation_cost_to_demandX (where X matches the customer_id in customer_demand.csv).

All index sets, parameters, and variable domains are defined exactly as in the current source data. No data is omitted or aggregated.
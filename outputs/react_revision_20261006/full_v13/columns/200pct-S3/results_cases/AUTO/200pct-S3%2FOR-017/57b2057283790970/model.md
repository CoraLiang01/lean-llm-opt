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
1. Demand satisfaction for each customer group:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity for each supplier:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (suppliers): All supplier_id values in supply_capacity.csv and transportation_costs.csv (source: file_1_view_0, file_2_view_0)
- $J$ (customers): All customer_id values in customer_demand.csv and all transportation_cost_to_C* columns in transportation_costs.csv (source: file_0_view_0, file_2_view_0)
- $d_j$: demand for customer $j$ from column demand in customer_demand.csv (source: file_0_view_0, columns customer_id, demand)
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv (source: file_1_view_0, columns supplier_id, supply_capacity)
- $c_{ij}$: transportation_cost_to_C* columns in transportation_costs.csv, with supplier_id as row and customer as column (source: file_2_view_0, columns supplier_id, transportation_cost_to_C1 ... transportation_cost_to_C10)

Variable:
- $x_{ij}$: quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$ (decision variable, continuous, nonnegative)

All index sets, parameters, and constraints are mapped directly to the current source data as described above. No data is omitted or aggregated.
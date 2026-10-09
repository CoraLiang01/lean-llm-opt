#### Mathematical Model

Let $I$ be the set of suppliers (from the supplier_id column of supply_capacity.csv and transportation_costs.csv), and $J$ be the set of customer groups (from the customer_id column of customer_demand.csv and the transportation_cost matrix columns).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (suppliers): All supplier_id values in supply_capacity.csv and transportation_costs.csv (source: file_1_view_0, file_2_view_0, column: supplier_id)
- $J$ (customer groups): All customer_id values in customer_demand.csv and all transportation_cost_to_C* columns in transportation_costs.csv (source: file_0_view_0, file_2_view_0, columns: customer_id, transportation_cost_to_C*)
- $d_j$: demand for customer group $j$ from customer_demand.csv (source: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply capacity for supplier $i$ from supply_capacity.csv (source: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ from transportation_costs.csv (source: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck mapped to $j$)

All index sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and coefficients.
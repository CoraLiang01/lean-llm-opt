#### Mathematical Model

Let $S$ be the set of suppliers (from supplier_id in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from customer_id in customer_demand.csv and transportation_costs.csv).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in S$ to customer $j \in C$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in C
\]
\[
\sum_{j \in C} x_{ij} \leq s_i \qquad \forall i \in S
\]
\[
x_{ij} \geq 0 \qquad \forall i \in S,\, j \in C
\]

#### Data Mapping

- $S$ (suppliers): All supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id) and transportation_costs.csv (table_id: file_2_view_0, column: supplier_id)
- $C$ (customers): All customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id) and as suffixes in transportation_costs.csv (table_id: file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C10)
- $d_j$: demand for customer $j$ from customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation_cost_to_Ck for supplier $i$ and customer $j$ from transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck where $k$ matches $j$)

All index sets and parameters are defined by the full set of records in the respective columns of the current CSV files. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.
#### Mathematical Model

Let:
- $I$ = set of suppliers, indexed by $i$ (from supplier_id in supply_capacity.csv and transportation_costs.csv)
- $J$ = set of customer groups, indexed by $j$ (from customer_id in customer_demand.csv and transportation_costs.csv)
- $d_j$ = demand of customer $j$ (from customer_demand.csv)
- $s_i$ = supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from supplier $i$ to customer $j$ (continuous)

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

- $I$ (suppliers): All supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id) and transportation_costs.csv (table_id: file_2_view_0, column: supplier_id)
- $J$ (customers): All customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id) and transportation_costs.csv (table_id: file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C10)
- $d_j$: demand for customer $j$ from customer_demand.csv (table_id: file_0_view_0, column: demand)
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0), entry in row with supplier_id $i$ and column transportation_cost_to_$j$ (where $j$ matches customer_id)
- $x_{ij}$: decision variable for each $(i,j)$ pair, continuous and nonnegative

Index sets $I$ and $J$ are defined by the full set of supplier_id and customer_id present in the respective CSVs and the transportation cost matrix. All constraints and parameters are mapped directly to the corresponding columns and rows as described above.
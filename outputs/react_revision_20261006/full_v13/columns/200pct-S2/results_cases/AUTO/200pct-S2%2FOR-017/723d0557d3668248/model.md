#### Mathematical Model

Let:
- $I$ = set of suppliers = $\{\text{S1}, \text{S2}, \ldots, \text{S10}\}$
- $J$ = set of customer groups = $\{\text{C1}, \text{C2}, \ldots, \text{C10}\}$
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer group $j \in J$ (continuous)
- $d_j$ = demand of customer group $j \in J$
- $s_i$ = supply capacity of supplier $i \in I$
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer group $j$

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

- $I$ (suppliers): All supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id)
- $J$ (customer groups): All customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand for customer $j$ from customer_demand.csv (table_id: file_0_view_0, column: demand, key: customer_id)
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity, key: supplier_id)
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0, row: supplier_id $i$, column: transportation_cost_to_$j$ where $j$ is customer_id)

All indices, parameters, and coefficients are mapped directly from the current CSV data as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.
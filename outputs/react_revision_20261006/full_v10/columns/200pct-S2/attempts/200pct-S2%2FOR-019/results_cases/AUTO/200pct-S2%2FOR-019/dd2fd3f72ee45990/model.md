##### Mathematical Model

Let
- $I$ = set of distribution centers (suppliers): $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$ = set of customer groups (demands): $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Parameters:
- $d_j$ = demand for customer group $j \in J$ (from customer_demand.csv)
- $s_i$ = supply capacity of distribution center $i \in I$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv)

Decision variables:
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (distribution centers): all unique values in column supplier_id of supply_capacity.csv (file_1_view_0)
- $J$ (customer groups): all unique values in column customer_id of customer_demand.csv (file_0_view_0)
- $d_j$: demand for $j$ from column demand in customer_demand.csv (file_0_view_0), indexed by customer_id
- $s_i$: supply_capacity for $i$ from column supply_capacity in supply_capacity.csv (file_1_view_0), indexed by supplier_id
- $c_{ij}$: transportation_costs.csv (file_2_view_0), row supplier_id $i$, column transportation_cost_to_${j}$ (where $j$ matches customer_id in demand), i.e., $c_{ij}$ is the value in the row with supplier_id $i$ and column transportation_cost_to_${j}$

Index sets, parameters, and all coefficients are mapped directly from the current CSV files as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.
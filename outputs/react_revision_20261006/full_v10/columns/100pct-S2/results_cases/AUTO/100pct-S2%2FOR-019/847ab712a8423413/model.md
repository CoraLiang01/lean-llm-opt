Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers): $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$

Let $J$ be the set of customer groups (demands): $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Parameters:
For all $i \in I$, $j \in J$:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column demand, row $j$)
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column supply_capacity, row $i$)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_$j$, row $i$)

Decision Variables:
For all $i \in I$, $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction (for all $j \in J$):
\[
\sum_{i \in I} x_{ij} \geq d_j
\]

2. Supply capacity (for all $i \in I$):
\[
\sum_{j \in J} x_{ij} \leq s_i
\]

3. Non-negativity (for all $i \in I$, $j \in J$):
\[
x_{ij} \geq 0
\]

Data Mapping

- $I$ (distribution centers): file_1_view_0, column supplier_id, all rows
- $J$ (customer groups): file_0_view_0, column customer_id, all rows
- $d_j$: file_0_view_0, column demand, row with customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity, row with supplier_id $i$
- $c_{ij}$: file_2_view_0, column transportation_cost_to_$j$, row with supplier_id $i$
- $x_{ij}$: decision variable for each $(i, j) \in I \times J$ (not present in data; to be determined by optimization)

All indices, parameters, and coefficients are mapped directly from the current CSV data as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.
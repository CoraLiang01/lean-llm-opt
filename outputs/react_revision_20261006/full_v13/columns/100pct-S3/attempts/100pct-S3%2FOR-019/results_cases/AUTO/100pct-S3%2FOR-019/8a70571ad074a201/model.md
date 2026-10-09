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
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): All unique values in column supplier_id of file_1_view_0 (supply_capacity.csv)
- $J$ (customer groups): All unique values in column customer_id of file_0_view_0 (customer_demand.csv)
- $d_j$: Value in column demand for customer_id $j$ in file_0_view_0 (customer_demand.csv)
- $s_i$: Value in column supply_capacity for supplier_id $i$ in file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: Value in column transportation_cost_to_${j}$ for supplier_id $i$ in file_2_view_0 (transportation_costs.csv), where $j$ matches the customer_id in file_0_view_0

- $x_{ij}$: Decision variable for each $(i, j) \in I \times J$ as defined above

Index sets, parameters, and all coefficients are mapped directly from the current CSV files as described. No data is omitted or aggregated.
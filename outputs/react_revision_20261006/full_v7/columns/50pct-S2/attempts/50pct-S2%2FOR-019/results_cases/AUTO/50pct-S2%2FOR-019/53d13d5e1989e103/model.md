##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$ (continuous).

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer group $j \in J$ (from customer_demand.csv)
- $s_i$: supply capacity of distribution center $i \in I$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from transportation_costs.csv)

##### Objective

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity:**  
   For each distribution center $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $d_j$ is the value in column "demand" for row with "customer_id" $j$ in table_id file_0_view_0 (customer_demand.csv).
- $s_i$ is the value in column "supply_capacity" for row with "supplier_id" $i$ in table_id file_1_view_0 (supply_capacity.csv).
- $c_{ij}$ is the value in column "transportation_cost_to_$j$" for row with "supplier_id" $i$ in table_id file_2_view_0 (transportation_costs.csv), where $j$ matches the demand column suffix (e.g., "demand1", ..., "demand8").

All indices, parameters, and coefficients are to be taken exactly as specified in the returned tables and relationships.
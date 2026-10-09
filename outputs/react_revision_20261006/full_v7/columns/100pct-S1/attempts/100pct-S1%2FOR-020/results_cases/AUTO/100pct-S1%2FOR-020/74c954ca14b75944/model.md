##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand at store $j$ (from customer_demand.csv)
- $s_i$: supply capacity at warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:** For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

##### Data Mapping

- $d_j$ is the value in column demand_units for customer_id $j$ in table_id file_0_view_0 (customer_demand.csv).
- $s_i$ is the value in column supply_capacity_units for supplier_id $i$ in table_id file_1_view_0 (supply_capacity.csv).
- $c_{ij}$ is the value in column transportation_cost_to_$j$ for supplier_id $i$ in table_id file_2_view_0 (transportation_costs.csv), where $j$ matches the customer_id in file_0_view_0.

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and mapped as above.
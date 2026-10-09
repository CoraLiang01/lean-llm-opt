##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (warehouses, from supply_capacity.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (stores, from customer_demand.csv)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): all "Unnamed: 0" values in supply_capacity.csv (table_id: file_1_view_0)
- $J$ (stores): all "customer" values in customer_demand.csv (table_id: file_0_view_0)
- $d_j$: "demand" column in customer_demand.csv, indexed by "customer" (table_id: file_0_view_0)
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "Unnamed: 0" (table_id: file_1_view_0)
- $c_{ij}$: value in transportation_costs.csv at row "Unnamed: 0" = $i$, column $j$ (table_id: file_2_view_0)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.
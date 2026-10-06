##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from warehouse $i$ to retail store $j$.

- $i \in I$ (warehouses), where $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $j \in J$ (retail stores), where $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of store $j$ (from `customer_demand.csv`)
- $s_i$: supply capacity of warehouse $i$ (from `supply_capacity.csv`)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `transportation_costs.csv`)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (warehouses): all values in column `"Unnamed: 0"` of table_id `file_1_view_0` and `file_2_view_0`
- $J$ (stores): all values in column `"customer"` of table_id `file_0_view_0` and columns `"C1"`–`"C10"` of table_id `file_2_view_0`
- $d_j$: value in column `"demand"` for row where `"customer" = j$ in table_id `file_0_view_0`
- $s_i$: value in column `"supply_capacity"` for row where `"Unnamed: 0" = i$ in table_id `file_1_view_0`
- $c_{ij}$: value in column $j$ for row where `"Unnamed: 0" = i$ in table_id `file_2_view_0`

All indices, parameters, and coefficients are bound exactly to the retrieved data as described above.
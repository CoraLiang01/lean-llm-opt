##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of retail stores (indexed by $j$).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from warehouse $i$ to store $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. **Demand satisfaction (for each store):**
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. **Supply capacity (for each warehouse):**
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): all values in column `"Unnamed: 0"` of table_id `"file_1_view_0"`  
- $J$ (stores): all values in column `"customer"` of table_id `"file_0_view_0"`
- $d_j$: demand for store $j$ from column `"demand"` in table_id `"file_0_view_0"`, keyed by `"customer"`
- $s_i$: supply capacity for warehouse $i$ from column `"supply_capacity"` in table_id `"file_1_view_0"`, keyed by `"Unnamed: 0"`
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ from table_id `"file_2_view_0"`, with warehouse $i$ as `"Unnamed: 0"` and store $j$ as column name

All indices, parameters, and coefficients are to be taken exactly as specified in the current Observation.
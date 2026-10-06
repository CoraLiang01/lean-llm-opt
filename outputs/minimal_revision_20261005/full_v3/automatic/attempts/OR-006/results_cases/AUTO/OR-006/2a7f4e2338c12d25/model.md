##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from warehouse $i$ to retail store $j$.

- $i \in I$ (warehouses), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $j \in J$ (retail stores), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each retail store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   where $d_j$ is the demand of store $j$.

2. **Warehouse supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is the supply capacity of warehouse $i$.

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Warehouses ($I$):**  
  From `supply_capacity.csv` (`file_1_view_0`), column `Unnamed: 0`  
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$

- **Retail Stores ($J$):**  
  From `customer_demand.csv` (`file_0_view_0`), column `customer`  
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

- **Demand ($d_j$):**  
  From `customer_demand.csv` (`file_0_view_0`), column `demand`  
  For each $j \in J$, $d_j = $ value in row where `customer` = $j$.

- **Supply Capacity ($s_i$):**  
  From `supply_capacity.csv` (`file_1_view_0`), column `supply_capacity`  
  For each $i \in I$, $s_i = $ value in row where `Unnamed: 0` = $i$.

- **Transportation Cost ($c_{ij}$):**  
  From `transportation_costs.csv` (`file_2_view_0`),  
  - Row identifier: `Unnamed: 0` (warehouse $i$)  
  - Column identifier: $j$ (store, e.g., `C1`, `C2`, ...)  
  For each $i \in I$, $j \in J$, $c_{ij} =$ value at row $i$, column $j$.

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data as described above.**
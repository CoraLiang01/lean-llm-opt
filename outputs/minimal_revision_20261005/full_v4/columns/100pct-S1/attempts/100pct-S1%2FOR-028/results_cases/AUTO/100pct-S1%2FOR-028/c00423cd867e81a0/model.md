##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, from column `"Warehouse (i)"` in `PotentialWarehouses_Costs.csv` (`file_0_view_0`).
- $J$: Set of stores, from column `"Store (j)"` in `Stores_Demands.csv` (`file_1_view_0`).
- $f_i$: Opening cost of warehouse $i$, from column `"Opening Cost (fi)"` in `file_0_view_0`.
- $u_i$: Capacity of warehouse $i$, from column `"Capacity (units)"` in `file_0_view_0`.
- $d_j$: Demand of store $j$, from column `"Demand (units, dj)"` in `file_1_view_0`.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from entry at row with `"Unnamed: 1" = Wi` and column `"Wj"` in `TransportationCost.csv` (`file_2_view_0`), where $i$ and $j$ are matched to warehouse and store indices.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All values in `"Warehouse (i)"`, `PotentialWarehouses_Costs.csv` (`file_0_view_0`)
- $J$: All values in `"Store (j)"`, `Stores_Demands.csv` (`file_1_view_0`)
- $f_i$: `"Opening Cost (fi)"` for warehouse $i$, `file_0_view_0`
- $u_i$: `"Capacity (units)"` for warehouse $i$, `file_0_view_0`
- $d_j$: `"Demand (units, dj)"` for store $j$, `file_1_view_0`
- $c_{ij}$: Entry in `TransportationCost.csv` (`file_2_view_0`), row with `"Unnamed: 1" = Wi`, column `"Wj"`, where $i$ and $j$ are warehouse and store indices (e.g., $i=1$ maps to `"W1"`, $j=1$ maps to `"W1"`, etc.)

---

**All index sets, parameters, and coefficients are defined directly from the CSV data as described above.**
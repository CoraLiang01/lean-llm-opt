##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $f_i$: Opening cost for warehouse $i$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.
- $u_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.

##### Index Sets

- $I$: Set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv.
- $J$: Set of stores, from column "Store (j)" in Stores_Demands.csv.

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

- $I$: All values in column `"Warehouse (i)"` of table_id `file_0_view_0` (PotentialWarehouses_Costs.csv)
- $J$: All values in column `"Store (j)"` of table_id `file_1_view_0` (Stores_Demands.csv)
- $f_i$: `"Opening Cost (fi)"` for warehouse $i$ in table_id `file_0_view_0`
- $u_i$: `"Capacity (units)"` for warehouse $i$ in table_id `file_0_view_0`
- $d_j$: `"Demand (units, dj)"` for store $j$ in table_id `file_1_view_0`
- $c_{ij}$: Value at row with `"Unnamed: 3" = Wk$"` (where $k$ is warehouse $i$) and column `"Wl"` (where $l$ is store $j$) in table_id `file_2_view_0` (TransportationCost.csv). That is, $c_{ij}$ is the entry in row $i$ and column $j$ of the warehouse-store cost matrix.

---

**All index sets, parameters, and constraints are defined directly from the provided CSV data.**
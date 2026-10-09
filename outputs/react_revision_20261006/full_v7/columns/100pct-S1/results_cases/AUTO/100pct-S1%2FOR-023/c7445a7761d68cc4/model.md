##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (the total demand).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from `file_1_view_0` and `file_2_view_0` column "Unnamed: 2".
- $J$: Set of stores, from `file_0_view_0` column "Customer" and `file_2_view_0` columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"].
- $d_j$: Demand of store $j$, from `file_0_view_0` column "demand".
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0` column "fixed_costs".
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` with rows indexed by "Unnamed: 2" (supplier) and columns as above (store).
- $M = \sum_{j \in J} d_j$, computed from all $d_j$ in `file_0_view_0`.

##### Data Mapping

- Suppliers ($I$):  
  - Table: `file_1_view_0` and `file_2_view_0`  
  - Column: "Unnamed: 2"
- Stores ($J$):  
  - Table: `file_0_view_0`  
  - Column: "Customer"  
  - Table: `file_2_view_0`  
  - Columns: ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"]
- Demand ($d_j$):  
  - Table: `file_0_view_0`  
  - Column: "demand"
- Fixed cost ($f_i$):  
  - Table: `file_1_view_0`  
  - Column: "fixed_costs"
- Transportation cost ($c_{ij}$):  
  - Table: `file_2_view_0`  
  - Rows: "Unnamed: 2" (supplier)  
  - Columns: ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] (store)
- $M$:  
  - Computed as the sum of all $d_j$ from `file_0_view_0` column "demand".

All index sets and parameters are defined directly from the CSV data as described above.
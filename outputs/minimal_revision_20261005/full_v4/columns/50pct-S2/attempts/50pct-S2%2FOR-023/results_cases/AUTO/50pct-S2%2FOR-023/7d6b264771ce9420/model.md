##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 1`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)

##### Parameters

- $d_j$: Demand at store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost to activate supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 0` = $i$, column = store name $j$)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:**  
   For every store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier Activation:**  
   For every supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i
   \]
   where $M = \sum_{j \in J} d_j$ (total demand; a valid upper bound since there are no explicit supplier capacity limits).

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (suppliers): `file_1_view_0`, column `Unnamed: 1`
- $J$ (stores): `file_0_view_0`, column `Customer`
- $d_j$: `file_0_view_0`, columns `Customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 1`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` = $i$, column = store name $j$ (mapping between store names in `file_0_view_0` and columns in `file_2_view_0` must be established as per data)
- $M = \sum_{j \in J} d_j$ (computed from all $d_j$ in `file_0_view_0`)

---

**Note:**  
- All index sets and parameters are defined directly from the CSV data as described above.
- The mapping between store names in the demand file and the transportation cost columns must be established according to the data dictionary or metadata. If store names differ, a mapping step is required.
##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers (indexed by $i$), corresponding to the rows in `fixed_cost.csv` and `transportation_costs.csv`.
- $J$: Set of stores (indexed by $j$), corresponding to the columns in `transportation_costs.csv` and the rows in `demand.csv`.

##### Parameters

- $f_i$: Fixed cost for activating supplier $i$.  
  Data mapping: `file_1_view_0`, column `fixed_costs`, row_id_mapping: `Unnamed: 0`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.  
  Data mapping: `file_2_view_0`, matrix with row_id_mapping: `Unnamed: 0` (suppliers), column_id_mapping: store names (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`).
- $d_j$: Demand at store $j$.  
  Data mapping: `file_0_view_0`, column `demand`, row_id_mapping: `Customer`.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- **Suppliers ($I$):**  
  Row identifiers from `file_1_view_0` and `file_2_view_0`, column `Unnamed: 0`.

- **Stores ($J$):**  
  Column identifiers from `file_2_view_0` (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`).

- **Demands ($d_j$):**  
  `file_0_view_0`, column `demand`, row_id_mapping: `Customer`.

- **Fixed costs ($f_i$):**  
  `file_1_view_0`, column `fixed_costs`, row_id_mapping: `Unnamed: 0`.

- **Transportation costs ($c_{ij}$):**  
  `file_2_view_0`, matrix with row_id_mapping: `Unnamed: 0` (suppliers), column_id_mapping: store names.

- **Big-M ($M$):**  
  $M = \sum_{j \in J} d_j$, using all demand values from `file_0_view_0`.

---

**Note:** All parameters and sets are mapped directly to the provided CSV data as described above. No values are enumerated; all identifiers and mappings are preserved as in the source files.
##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation (no conditional capacity):**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 2`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)
- $d_j$: Demand at store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 2` for supplier, column header for store)

##### Data Mapping

- $I$: All values in `file_1_view_0`, column `Unnamed: 2`
- $J$: All values in `file_0_view_0`, column `Customer`
- $d_j$: `file_0_view_0`, columns `Customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 2`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, rows indexed by `Unnamed: 2` (supplier), columns named for each store in $J$ (e.g., `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`)

**Note:** The mapping between store names in the demand file (`Customer`) and the transportation cost columns must be established by the user if not directly aligned. All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.
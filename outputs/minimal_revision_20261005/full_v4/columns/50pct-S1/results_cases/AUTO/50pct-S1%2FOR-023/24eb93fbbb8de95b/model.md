##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers (from `file_1_view_0` and `file_2_view_0` row identifiers).
- $J$: Set of stores (from `file_0_view_0` Customer column and `file_2_view_0` column identifiers).

##### Parameters

- $d_j$: Demand at store $j$ (from `file_0_view_0`, column: demand, indexed by Customer).
- $f_i$: Fixed cost to activate supplier $i$ (from `file_1_view_0`, column: fixed_costs, indexed by Unnamed: 1).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row: Unnamed: 0, column: store name).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$ (Suppliers): All unique values in `file_1_view_0` column `Unnamed: 1` and `file_2_view_0` rows `Unnamed: 0`.
- $J$ (Stores): All unique values in `file_0_view_0` column `Customer` and `file_2_view_0` columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- $d_j$: `file_0_view_0`, column `demand`, indexed by `Customer`.
- $f_i$: `file_1_view_0`, column `fixed_costs`, indexed by `Unnamed: 1`.
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` (supplier), column (store).
- $M$: $\sum_{j \in J} d_j$ (sum over all `demand` in `file_0_view_0`).

**Note:** All index sets and parameters are to be taken exactly as present in the referenced columns and rows of the CSV files. No values are to be omitted or abbreviated.
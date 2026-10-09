##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
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
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (the total demand across all stores).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `fixed_cost.csv` and rows of `transportation_costs.csv`)
- $J$: Set of stores (from columns of `transportation_costs.csv`)
- $d_j$: Demand of store $j$ (from `demand.csv`)
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`)
- $M$: Total demand, $M = \sum_{j \in J} d_j$

##### Data Mapping

- $I$: All values in column `Unnamed: 0` of table_id `file_1_view_0` and `file_2_view_0`
- $J$: All values in column `Customer` of table_id `file_0_view_0` (demand.csv) and columns (excluding `Unnamed: 0`) of table_id `file_2_view_0` (transportation_costs.csv)
- $d_j$: Column `demand` in table_id `file_0_view_0`, indexed by `Customer`
- $f_i$: Column `fixed_costs` in table_id `file_1_view_0`, indexed by `Unnamed: 0`
- $c_{ij}$: Table_id `file_2_view_0`, rows indexed by `Unnamed: 0` (supplier), columns indexed by store names (see $J$)
- $M$: $M = \sum_{j \in J} d_j$ (sum over all $d_j$ from `file_0_view_0`)

**Note:** The mapping between store names in `demand.csv` and columns in `transportation_costs.csv` must be established by the user if not directly aligned. All index sets and parameters are defined by the full set of unique identifiers in the respective columns of the source tables.
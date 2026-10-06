##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to activate supplier $i$.

##### Constraints

1. **Demand Satisfaction:**  
   Each store's demand must be fully met:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand of store $j$.

2. **Supplier Activation:**  
   No shipments from inactive suppliers:
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (facilities), indexed by $i$.
- $J$: Set of stores (customers), indexed by $j$.

##### Data Mapping

- $I$: All unique values in `file_1_view_0` column `Unnamed: 2` (supplier/facility names).
- $J$: All unique values in `file_0_view_0` column `Customer` (store/customer names).
- $d_j$: From `file_0_view_0`, column `demand`, keyed by `Customer`.
- $f_i$: From `file_1_view_0`, column `fixed_costs`, keyed by `Unnamed: 2`.
- $c_{ij}$: From `file_2_view_0`, row `Unnamed: 2` (supplier/facility), column header matching $j$ (store/customer name).
- $M$: $M = \sum_{j \in J} d_j$ (sum of all values in `file_0_view_0` column `demand`).

##### Notes

- All index sets and parameters are defined directly from the CSV data as described above.
- The transportation cost matrix $c_{ij}$ is constructed by matching each supplier (row `Unnamed: 2` in `file_2_view_0`) to each store (column header in `file_2_view_0` matching a value in `Customer` from `file_0_view_0`).
- If a store name in $J$ does not exactly match a column in `file_2_view_0`, column mapping must be resolved by the user or data provider.
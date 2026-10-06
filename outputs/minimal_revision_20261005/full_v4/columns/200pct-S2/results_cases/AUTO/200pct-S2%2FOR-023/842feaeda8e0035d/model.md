##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store Demand Satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand must be fully met.)

2. **Supplier Activation Constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   (No shipments from inactive suppliers; $M = \sum_{j \in J} d_j$ is a valid upper bound.)

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility locations), from `file_1_view_0` and `file_2_view_0` column `"Unnamed: 3"` and `"Unnamed: 4"`.
- $J$: Set of stores (customers), from `file_0_view_0` column `"Customer"` and `file_2_view_0` columns `["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"]`.
- $d_j$: Demand for store $j$, from `file_0_view_0` column `"demand"`.
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0` column `"fixed_costs"`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` with rows indexed by `"Unnamed: 4"` and columns by store names.
- $M = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$ (Suppliers):  
  - Source: `file_1_view_0`, column `"Unnamed: 3"`  
  - Source: `file_2_view_0`, column `"Unnamed: 4"` (row index for cost matrix)
- $J$ (Stores):  
  - Source: `file_0_view_0`, column `"Customer"`  
  - Source: `file_2_view_0`, columns `["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"]`
- $d_j$:  
  - Source: `file_0_view_0`, column `"demand"`, indexed by `"Customer"`
- $f_i$:  
  - Source: `file_1_view_0`, column `"fixed_costs"`, indexed by `"Unnamed: 3"`
- $c_{ij}$:  
  - Source: `file_2_view_0`, value at row with `"Unnamed: 4" = i`, column $j$
- $M$:  
  - $M = \sum_{j \in J} d_j$, using all $d_j$ from `file_0_view_0`

##### Notes

- All suppliers and stores present in the data are included in $I$ and $J$.
- The cost matrix $c_{ij}$ is defined by matching supplier names in `"Unnamed: 4"` (rows) and store names in the column headers of `file_2_view_0`.
- The model ensures that each store's demand is met, suppliers are only activated if they ship, and total cost (fixed + transportation) is minimized.
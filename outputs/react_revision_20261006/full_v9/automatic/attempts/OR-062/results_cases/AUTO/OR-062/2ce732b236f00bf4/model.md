##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
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

2. **Supplier Activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (A supplier can only ship if activated. $M_i$ is a sufficiently large constant, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `file_1_view_0` "Unnamed: 0")
- $J$: Set of stores (from `file_2_view_0` columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT")
- $d_j$: Demand for store $j$ (from `file_0_view_0`, column "demand", indexed by "Customer")
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 0")
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row "Unnamed: 0" and columns as above)
- $M_i$: Big-M constant for each supplier $i$ (set as $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- Suppliers $I$: `file_1_view_0` column "Unnamed: 0"
- Stores $J$: `file_2_view_0` columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- Demand $d_j$: `file_0_view_0` columns "Customer", "demand"
- Fixed cost $f_i$: `file_1_view_0` columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: `file_2_view_0` rows "Unnamed: 0", columns as above
- $M_i$: $M_i = \sum_{j \in J} d_j$ (sum over all store demands from `file_0_view_0`)

No additional constraints or capacity limits are specified beyond those above. All index sets and parameters are defined directly from the CSV data.
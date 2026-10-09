##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

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
   x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $D_j$ is the demand of store $j$ (from data), ensuring $x_{ij}=0$ if $y_i=0$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (from fixed_cost.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J$: Set of stores (from demand.csv, column "customer", table_id: file_0_view_0)

##### Parameters and Data Mapping

- $d_j$: Demand of store $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$: Fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, row "Unnamed: 0" for $i$, columns $J$ for $j$, table_id: file_2_view_0)

##### Data Mapping

- Suppliers $I$: file_1_view_0, column "Unnamed: 0"
- Stores $J$: file_0_view_0, column "customer"
- Demand $d_j$: file_0_view_0, columns "customer", "demand"
- Fixed cost $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), columns $J$ (store)
- Activation constraint upper bound $D_j$: file_0_view_0, column "demand"

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.
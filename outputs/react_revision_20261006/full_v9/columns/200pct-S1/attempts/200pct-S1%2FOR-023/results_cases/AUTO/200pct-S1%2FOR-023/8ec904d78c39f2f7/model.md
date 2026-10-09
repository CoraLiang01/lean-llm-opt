##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

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
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (the total demand).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from fixed_cost.csv and transportation_costs.csv rows).
- $J$: Set of stores (from demand.csv and transportation_costs.csv columns).
- $d_j$: Demand of store $j$ (from demand.csv, column "demand", table_id: file_0_view_0).
- $f_i$: Fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, rows indexed by supplier, columns by store).
- $M$: $\sum_{j \in J} d_j$ (sum of all store demands).

##### Data Mapping

- Suppliers $I$:  
  - Names: file_1_view_0, column "Unnamed: 3" (fixed_cost.csv)  
  - Also: file_2_view_0, column "Unnamed: 4" (transportation_costs.csv, row labels)
- Stores $J$:  
  - Names: file_0_view_0, column "Customer" (demand.csv)  
  - Also: file_2_view_0, columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] (transportation_costs.csv, column labels)
- Demand $d_j$: file_0_view_0, column "demand"
- Fixed cost $f_i$: file_1_view_0, column "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, matrix with rows indexed by "Unnamed: 4" (supplier), columns by store names above
- $M$: $\sum_{j \in J} d_j$ (sum over file_0_view_0, column "demand")

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.
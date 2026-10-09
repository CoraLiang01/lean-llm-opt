##### Decision Variables

- $x_{ij} \geq 0$: quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

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
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" and "file_2_view_0".
- $J$: set of stores, from column "customer" in table_id "file_0_view_0" and columns "C1"–"C6" in table_id "file_2_view_0".

##### Parameters and Data Mapping

- $d_j$: demand of store $j$, from column "demand" in table_id "file_0_view_0", indexed by "customer".
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0", indexed by "Unnamed: 0".
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$, from table_id "file_2_view_0", row "Unnamed: 0" (supplier), column $j$ ("C1"–"C6").
- $M = \sum_{j \in J} d_j$, with $d_j$ as above.

##### Data Mapping

- Suppliers $I$: table_id "file_1_view_0", column "Unnamed: 0"
- Stores $J$: table_id "file_0_view_0", column "customer"
- Demand $d_j$: table_id "file_0_view_0", columns "customer", "demand"
- Fixed cost $f_i$: table_id "file_1_view_0", columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: table_id "file_2_view_0", row "Unnamed: 0" (supplier), columns "C1"–"C6" (stores)
- $M$: sum of all $d_j$ from table_id "file_0_view_0", column "demand"

No supplier capacity limits are imposed beyond the activation logic. All indices and parameters are mapped directly from the CSV sources as described.
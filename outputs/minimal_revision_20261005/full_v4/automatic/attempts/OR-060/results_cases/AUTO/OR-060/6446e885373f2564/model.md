##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: Set of supermarkets, from column "customer" in table_id file_0_view_0 and columns "C1",...,"C12" in file_2_view_0.
- $d_j$: Demand of supermarket $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: Fixed cost for opening supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to supermarket $j$, from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), ensuring that no goods are shipped from closed suppliers.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All supplier IDs from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: All supermarket IDs from column "customer" in table_id file_0_view_0 and columns "C1",...,"C12" in file_2_view_0.
- $d_j$: From table_id file_0_view_0, column "demand", indexed by "customer".
- $f_i$: From table_id file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0".
- $c_{ij}$: From table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.
- $M$: $M = \sum_{j \in J} d_j$, with $d_j$ as above.
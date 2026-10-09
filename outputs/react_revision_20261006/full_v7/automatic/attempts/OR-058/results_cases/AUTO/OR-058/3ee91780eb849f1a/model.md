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
   x_{ij} \leq M_{ij} y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $M_{ij}$ is any valid upper bound on $x_{ij}$ (e.g., $M_{ij} = d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: Set of stores, from column "customer" in table_id file_0_view_0 and columns "C1"..."C6" in file_2_view_0.
- $d_j$: Demand of store $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from table_id file_2_view_0, row "Unnamed: 0" (supplier), column $j$.
- $M_{ij} = d_j$.

##### Data Mapping

- $I$: All values in "Unnamed: 0" column of file_1_view_0 and file_2_view_0.
- $J$: All values in "customer" column of file_0_view_0 and columns "C1"..."C6" of file_2_view_0.
- $d_j$: "demand" column in file_0_view_0, indexed by "customer".
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), column $j$ (store).
- $M_{ij} = d_j$ for each $i,j$.

No additional capacity or side constraints are imposed beyond those above.
##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   x_{ij} \leq D_j y_i, \quad \forall i \in I,\, \forall j \in J
   \]
   where $D_j$ is the demand of branch $j$ (ensures $x_{ij}=0$ if $y_i=0$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0"
- $J$: set of branches, from column "customer" in table_id "file_0_view_0"

##### Parameter Mapping

- $d_j$: demand of branch $j$, from column "demand" in table_id "file_0_view_0"
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0"
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from table_id "file_2_view_0", row "Unnamed: 0" = $i$, column $j$
- $D_j$: demand of branch $j$, from column "demand" in table_id "file_0_view_0"

##### Data Mapping

- Suppliers $I$: all values in "Unnamed: 0" of "file_1_view_0"
- Branches $J$: all values in "customer" of "file_0_view_0"
- $d_j$: "demand" in "file_0_view_0" for branch $j$
- $f_i$: "fixed_costs" in "file_1_view_0" for supplier $i$
- $c_{ij}$: "file_2_view_0", row "Unnamed: 0" = $i$, column $j$
- $D_j$: "demand" in "file_0_view_0" for branch $j$

All index sets and parameters are defined by the full set of entities in the current Observation.
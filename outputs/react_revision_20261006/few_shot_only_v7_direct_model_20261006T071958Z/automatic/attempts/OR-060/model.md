##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each supermarket $j$ must receive exactly its demand $d_j$.)

2. **Supplier can only ship if open:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (If supplier $i$ is not open ($y_i=0$), it cannot ship any goods. $M$ is a sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameter Mapping

- $I$: set of suppliers, from column `"Unnamed: 0"` in `file_1_view_0` (`fixed_cost.csv`)
- $J$: set of supermarkets, from column `"customer"` in `file_0_view_0` (`demand.csv`)
- $d_j$: demand of supermarket $j$, from column `"demand"` in `file_0_view_0` (`demand.csv`)
- $f_i$: fixed cost for supplier $i$, from column `"fixed_costs"` in `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from column $j$ in row $i$ of `file_2_view_0` (`transportation_costs.csv`)
- $M = \sum_{j \in J} d_j$ (sum of all supermarket demands, computed from `file_0_view_0`)

##### Data Mapping

- $I$: `"Unnamed: 0"` in `file_1_view_0` (`fixed_cost.csv`)
- $J$: `"customer"` in `file_0_view_0` (`demand.csv`)
- $d_j$: `"demand"` in `file_0_view_0` (`demand.csv`)
- $f_i$: `"fixed_costs"` in `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: column $j$ in row $i$ of `file_2_view_0` (`transportation_costs.csv`)
- $M$: $\sum_{j \in J} d_j$ from `"demand"` in `file_0_view_0` (`demand.csv`)
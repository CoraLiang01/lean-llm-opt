##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous), for all suppliers $i$ and customers $j$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. **Supplier activation:**  
   $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$  
   where $M_i = \sum_{j \in J} d_j$ (a valid upper bound, since there are no explicit supplier capacity limits).

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 1" in `fixed_cost.csv` ([file_1_view_0])
- $J$: set of customers, from column "Customer" in `demand.csv` ([file_0_view_0])
- $d_j$: demand of customer $j$, from column "demand" in `demand.csv` ([file_0_view_0])
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv` ([file_1_view_0])
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` ([file_2_view_0]), with supplier $i$ as row "Unnamed: 0" and customer $j$ as column header (see Data Mapping below).

##### Data Mapping

- $I$: All values in [file_1_view_0], column "Unnamed: 1"
- $J$: All values in [file_0_view_0], column "Customer"
- $d_j$: [file_0_view_0], column "demand", keyed by "Customer"
- $f_i$: [file_1_view_0], column "fixed_costs", keyed by "Unnamed: 1"
- $c_{ij}$: [file_2_view_0], row "Unnamed: 0" (supplier $i$), column $j$ (customer location name, as in column headers)
- $M_i$: $\sum_{j \in J} d_j$ (sum over all customer demands)

**Note:** The transportation cost matrix [file_2_view_0] uses supplier names as row "Unnamed: 0" and customer location names as columns. Ensure mapping between customer names in $J$ and the corresponding columns in the transportation cost matrix.

##### Complete Model

$\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}$

**All index sets and parameters are defined above and mapped to their exact CSV sources.**
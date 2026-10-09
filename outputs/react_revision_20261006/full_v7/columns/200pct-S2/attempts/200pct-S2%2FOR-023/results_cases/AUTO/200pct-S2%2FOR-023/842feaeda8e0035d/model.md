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
   (Each store's demand must be fully met.)

2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   (No shipments from inactive suppliers; $M$ is a sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 3" in `fixed_cost.csv` ([table_id: file_1_view_0]).
- $J$: Set of stores, from column "Customer" in `demand.csv` ([table_id: file_0_view_0]).
- $d_j$: Demand of store $j$, from column "demand" in `demand.csv` ([table_id: file_0_view_0]).
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv` ([table_id: file_1_view_0]).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `transportation_costs.csv` ([table_id: file_2_view_0]), with supplier $i$ identified by "Unnamed: 4" and store $j$ by the corresponding column header.
- $M$: $M = \sum_{j \in J} d_j$ (sum of all store demands).

##### Data Mapping

- $I$: All unique values in [file_1_view_0, column "Unnamed: 3"].
- $J$: All unique values in [file_0_view_0, column "Customer"].
- $d_j$: [file_0_view_0, columns "Customer", "demand"].
- $f_i$: [file_1_view_0, columns "Unnamed: 3", "fixed_costs"].
- $c_{ij}$: [file_2_view_0, rows indexed by "Unnamed: 4" (supplier), columns indexed by store names].
- $M$: $\sum_{j \in J} d_j$ (sum over [file_0_view_0, column "demand"]).

No additional constraints are imposed unless specified in the data. All index sets and parameters are defined directly from the current CSV data.
##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Supermarket demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation (no shipment from inactive suppliers):
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$.
3. Variable domains:
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: set of supermarkets, from column "customer" in table_id file_0_view_0 and columns in file_2_view_0 (excluding "Unnamed: 0").

##### Parameters and Data Mapping

- $d_j$: demand of supermarket $j$, from column "demand" in table_id file_0_view_0, indexed by "customer".
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$, from table_id file_2_view_0, rows indexed by "Unnamed: 0" (supplier), columns by supermarket IDs.
- $M = \sum_{j \in J} d_j$, with $d_j$ as above.

##### Data Mapping

- Supermarkets $J$: file_0_view_0, column "customer"
- Suppliers $I$: file_1_view_0, column "Unnamed: 0"
- Demand $d_j$: file_0_view_0, columns "customer", "demand"
- Fixed cost $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, rows "Unnamed: 0", columns "C1", "C2" (supermarkets)
- $M$: sum over $d_j$ from file_0_view_0

All index sets and parameters are defined directly from the current CSV data as described above.
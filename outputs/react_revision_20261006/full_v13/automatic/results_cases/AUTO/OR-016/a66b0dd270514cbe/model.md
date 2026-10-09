#### Symbolic Mathematical Model

Let:
- $I$ = set of all products (indexed by $i$), as defined by all "Product Name" entries in table_id file_0_view_0.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from "Revenue")
    - $d_i$ = demand for product $i$ (parameter from "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All "Product Name" entries in table_id file_0_view_0 (RetailSalesDataset.csv, column "Product Name")
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $s_i$: "Initial Inventory" column in table_id file_0_view_0
- Variable $x_i$: Decision variable for each $i \in I$ (product)

All parameters are mapped directly from the specified columns in table_id file_0_view_0. No additional constraints or bounds are imposed beyond those described above.
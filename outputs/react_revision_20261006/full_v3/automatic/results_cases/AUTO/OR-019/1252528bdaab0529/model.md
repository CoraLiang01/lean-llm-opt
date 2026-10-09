#### Sets
Let $\mathcal{I}$ be the set of all products classified under ‘27in’ in the dataset.

#### Parameters
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

#### Decision Variables
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

#### Constraints
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping
- Set $\mathcal{I}$: All rows in table_id: file_0_view_0 where "Product Name" has prefix "27in"
- Parameter $A_i$: "Revenue" column, table_id: file_0_view_0
- Parameter $d_i$: "Demand" column, table_id: file_0_view_0
- Parameter $I_i$: "Initial Inventory" column, table_id: file_0_view_0
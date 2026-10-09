#### Sets
Let $\mathcal{I}$ be the set of all products classified under ‘Books’ in the dataset.

#### Parameters
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

#### Decision Variables
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

#### Constraints
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in \mathcal{I} \\
& x_i \leq I_i, && \forall i \in \mathcal{I} \\
& x_i \geq 0, && \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}, && \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping
- Table: file_0_view_0 (from DifferentStoreSales.csv)
    - Index set $\mathcal{I}$: All rows where "Product_Name" has prefix "Books"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $I_i$: column "Initial Inventory"
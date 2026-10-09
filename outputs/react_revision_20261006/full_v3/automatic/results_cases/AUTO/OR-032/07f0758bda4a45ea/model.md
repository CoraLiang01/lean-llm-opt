#### Sets
Let $\mathcal{I}$ be the set of all products classified under ‘Books’ in table_id file_0_view_0, column Product_Name.

#### Parameters
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from table_id file_0_view_0, column Revenue)
- $d_i$: Demand for product $i$ (from table_id file_0_view_0, column Demand)
- $I_i$: Initial Inventory for product $i$ (from table_id file_0_view_0, column Initial Inventory)

#### Decision Variables
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_+$

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
- Set $\mathcal{I}$: All records in table_id file_0_view_0 with Product_Name prefix "Books"
- Parameter $A_i$: table_id file_0_view_0, column Revenue
- Parameter $d_i$: table_id file_0_view_0, column Demand
- Parameter $I_i$: table_id file_0_view_0, column Initial Inventory
##### Sets
Let $\mathcal{I}$ be the set of all car models with Product Name prefix ‘FDK57’ as returned from table_id file_0_view_0.

##### Parameters
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of car model $i$ (from column ‘Revenue’)
- $d_i$: Demand for car model $i$ (from column ‘Demand’)
- $I_i$: Initial Inventory for car model $i$ (from column ‘Initial Inventory’)

##### Decision Variables
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of car model $i$ to fulfill, $x_i \in \mathbb{Z}_+$

##### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

##### Constraints
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in \mathcal{I} \\
& x_i \leq I_i, && \forall i \in \mathcal{I} \\
& x_i \geq 0, && \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}, && \forall i \in \mathcal{I}
\end{align*}
\]

##### Data Mapping
- Set $\mathcal{I}$: All records in table_id file_0_view_0 with ‘Product Name’ prefix ‘FDK57’
- $A_i$: file_0_view_0, column ‘Revenue’
- $d_i$: file_0_view_0, column ‘Demand’
- $I_i$: file_0_view_0, column ‘Initial Inventory’
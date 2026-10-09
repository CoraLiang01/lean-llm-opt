Mathematical Optimization Model

Index Sets:
- $I$: Set of all products in the dataset with "Organ" in their 'Sub Category' name.

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column 'Revenue', table_id: file_0_view_0)
- $d_i$: Demand for product $i \in I$ (from column 'Demand', table_id: file_0_view_0)
- $I_i$: Initial inventory for product $i \in I$ (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq I_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

Data Mapping:
- Index set $I$ is defined as all rows in table_id file_0_view_0 where 'Sub Category' contains "Organ".
- Parameter $A_i$ is mapped from column 'Revenue' in table_id file_0_view_0.
- Parameter $d_i$ is mapped from column 'Demand' in table_id file_0_view_0.
- Parameter $I_i$ is mapped from column 'Initial Inventory' in table_id file_0_view_0.
- Decision variable $x_i$ is defined for each $i \in I$.

No additional constraints or bounds are imposed beyond those specified above.
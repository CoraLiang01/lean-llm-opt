#### Abstract Mathematical Model

**Index Set:**  
Let $\mathcal{B}$ be the set of all products where the value in column `Product_Name` begins with "Books".

**Parameters:**  
For each $i \in \mathcal{B}$:
- $A_i$: revenue per unit of product $i$ (from column `Revenue`)
- $d_i$: total demand for product $i$ (from column `Demand`)
- $I_i$: initial inventory for product $i$ (from column `Initial Inventory`)

**Decision Variables:**  
For each $i \in \mathcal{B}$:
- $x_i \in \mathbb{Z}_+$: number of units of product $i$ to fulfill (non-negative integer)

**Objective:**  
\[
\max \sum_{i \in \mathcal{B}} A_i x_i
\]

**Constraints:**  
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in \mathcal{B} \quad \text{(Demand constraint)} \\
& x_i \leq I_i, && \forall i \in \mathcal{B} \quad \text{(Inventory constraint)} \\
& x_i \geq 0, && \forall i \in \mathcal{B} \quad \text{(Non-negativity and integrality)}
\end{align*}
\]

#### Data Mapping

- **Index set $\mathcal{B}$:** All records in table_id `file_0_view_0` where `Product_Name` starts with "Books"
- **Parameter $A_i$:** `Revenue` column in table_id `file_0_view_0`
- **Parameter $d_i$:** `Demand` column in table_id `file_0_view_0`
- **Parameter $I_i$:** `Initial Inventory` column in table_id `file_0_view_0`
- **Decision variable $x_i$:** Defined for each $i \in \mathcal{B}$

All data is sourced from table_id `file_0_view_0` with columns as specified.
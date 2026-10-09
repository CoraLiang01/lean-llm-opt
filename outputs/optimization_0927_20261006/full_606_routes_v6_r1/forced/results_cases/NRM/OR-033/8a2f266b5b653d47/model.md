Index Sets:
- $I$: Set of all products classified under ‘Baby’ (from column "Product Name" in table_id file_0_view_0).

Parameters:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand" in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq I_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
\]

Data Mapping:
- Table: file_0_view_0 (EuropeSalesRecords.csv)
    - Index set $I$: All rows where "Product Name" has prefix "Baby"
    - $A_i$: "Revenue"
    - $d_i$: "Demand"
    - $I_i$: "Initial Inventory"
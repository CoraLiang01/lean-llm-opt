Mathematical Model

Sets:
- $I$: Set of products classified under ‘Aalop’ (indexed by $i$; from column "Product Name" in table_id file_0_view_0).

Parameters:
- $r_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0).
- $d_i$: Demand for product $i$ during the sales horizon ("Demand", file_0_view_0).
- $s_i$: Initial inventory of product $i$ ("Initial Inventory", file_0_view_0).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 \text{ and integer}, && \forall i \in I
\end{align*}
\]

Data Mapping

- $I$: All records in file_0_view_0, column "Product Name", filtered by prefix "Aalop".
- $r_i$: file_0_view_0, column "Revenue", key: "Product Name".
- $d_i$: file_0_view_0, column "Demand", key: "Product Name".
- $s_i$: file_0_view_0, column "Initial Inventory", key: "Product Name".
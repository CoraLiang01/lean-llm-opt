ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of products classified as ‘27in’ (indexed by $i$; see Data Mapping).

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $d_i$: Demand for product $i$ (from Demand column).
- $s_i$: Initial Inventory of product $i$ (from Initial Inventory column).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \in \mathbb{Z}_{\geq 0}, && \forall i \in I \\
\end{align*}
\]

Data Mapping:
- $I$: All rows in Salesorders.csv where Product Name starts with "27in".
- $r_i$: Salesorders.csv, column Revenue, table_id: file_0_view_0, key: Product Name.
- $d_i$: Salesorders.csv, column Demand, table_id: file_0_view_0, key: Product Name.
- $s_i$: Salesorders.csv, column Initial Inventory, table_id: file_0_view_0, key: Product Name.
Mathematical Model

Sets:
- $I$: Set of dairy products, indexed by $i$ (from all Full_Product_Name in file_0_view_0)

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue, file_0_view_0)
- $d_i$: Demand for product $i$ (from Demand, file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (from Initial Inventory, file_0_view_0)

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq s_i, && \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, && \forall i \in I
\end{align*}
\]

Data Mapping

- $I$: All Full_Product_Name in table_id file_0_view_0, column Full_Product_Name
- $r_i$: Revenue, table_id file_0_view_0, column Revenue, keyed by Full_Product_Name
- $d_i$: Demand, table_id file_0_view_0, column Demand, keyed by Full_Product_Name
- $s_i$: Initial Inventory, table_id file_0_view_0, column Initial Inventory, keyed by Full_Product_Name
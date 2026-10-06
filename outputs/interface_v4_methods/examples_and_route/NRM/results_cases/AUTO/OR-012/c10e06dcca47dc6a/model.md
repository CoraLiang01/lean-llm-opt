ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of products, indexed by $i$ (see Data Mapping for Product Name).

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $d_i$: Demand for product $i$ (from Demand column).
- $s_i$: Initial Inventory for product $i$ (from Initial Inventory column).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
 Domain: $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
- Inventory and Demand Fulfillment Constraints:
\[
0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
\]

Data Mapping:
- $I$: All records in OnlineSalesDataset.csv, column Product Name, table_id=file_0_view_0
- $r_i$: OnlineSalesDataset.csv, column Revenue, table_id=file_0_view_0
- $d_i$: OnlineSalesDataset.csv, column Demand, table_id=file_0_view_0
- $s_i$: OnlineSalesDataset.csv, column Initial Inventory, table_id=file_0_view_0

Variable mapping:
- $x_i$ is indexed by Product Name from OnlineSalesDataset.csv, table_id=file_0_view_0

All 119 product records are included, preserving source order and identifiers.
Mathematical Model

Index Sets:
- $I$: Set of products, indexed by $i$ (from "Product Name" in file_0_view_0).

Parameters:
- $r_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0).
- $d_i$: Demand for product $i$ ("Demand", file_0_view_0).
- $s_i$: Initial inventory for product $i$ ("Initial Inventory", file_0_view_0).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in I
\]
\[
x_i \leq s_i \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping

- $I$: All rows in table_id file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$ (product in file_0_view_0, "Product Name")
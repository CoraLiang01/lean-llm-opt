Mathematical Model

Index Sets:
- $I$: Set of products, indexed by $i$ (corresponds to "Product Name" in file_0_view_0)

Parameters:
- $r_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0)
- $d_i$: Demand for product $i$ ("Demand", file_0_view_0)
- $s_i$: Initial inventory of product $i$ ("Initial Inventory", file_0_view_0)

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill for customer purchases ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Inventory and demand fulfillment constraints:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

- Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- $I$: All rows in file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$ (product "Product Name")
Mathematical Model

Sets:
- $I$: set of baked goods, indexed by $i$ (from all "Product Name" in file_0_view_0)

Parameters:
- $r_i$: revenue per unit of baked good $i$ ("Revenue", file_0_view_0)
- $d_i$: demand for baked good $i$ ("Demand", file_0_view_0)
- $s_i$: initial inventory of baked good $i$ ("Initial Inventory", file_0_view_0)

Decision Variables:
- $x_i$: quantity of baked good $i$ to fulfill (integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Constraints:
1. Inventory and demand fulfillment:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

2. Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- $I$: All rows in table_id file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: decision variable for each $i \in I$ (baked good)

All parameters are mapped directly from the specified columns in file_0_view_0.
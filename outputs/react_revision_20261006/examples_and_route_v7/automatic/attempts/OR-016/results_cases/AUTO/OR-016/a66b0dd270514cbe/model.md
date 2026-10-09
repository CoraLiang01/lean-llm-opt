##### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (from all "Product Name" in file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0)
- $d_i$: Demand for product $i$ ("Demand", file_0_view_0)
- $s_i$: Initial inventory for product $i$ ("Initial Inventory", file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory limit for each product:
   $$
   x_i \leq s_i \quad \forall i \in I
   $$
2. Demand limit for each product:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

##### Data Mapping

- $I$: All products from file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$

All parameters are mapped directly from file_0_view_0 as specified.
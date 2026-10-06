#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from WomenClothingEcommerceSalesData.csv, column "Product Name")
- $r_i$ = revenue per unit of product $i$ ("Revenue", table_id: file_0_view_0)
- $d_i$ = demand for product $i$ ("Demand", table_id: file_0_view_0)
- $s_i$ = initial inventory of product $i$ ("Initial Inventory", table_id: file_0_view_0)
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Parameters:**
- $r_i$: file_0_view_0, column "Revenue", key "Product Name"
- $d_i$: file_0_view_0, column "Demand", key "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", key "Product Name"

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory constraint:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

#### Data Mapping

- $I$: file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", key "Product Name"
- $d_i$: file_0_view_0, column "Demand", key "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", key "Product Name"

All parameters and index sets are derived directly from the returned table, preserving original order and identifiers.
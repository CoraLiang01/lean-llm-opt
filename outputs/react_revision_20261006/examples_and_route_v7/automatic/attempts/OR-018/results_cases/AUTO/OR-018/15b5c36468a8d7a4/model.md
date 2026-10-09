##### Mathematical Model

Let $I$ be the set of products classified under ‘Baby’ (from column "Product Name" with prefix "Baby") in table_id file_0_view_0.

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (column "Revenue", table_id file_0_view_0)
- $d_i$: Demand for product $i$ (column "Demand", table_id file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (column "Initial Inventory", table_id file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 where "Product Name" starts with "Baby"
- $r_i$: "Revenue" column, table_id file_0_view_0
- $d_i$: "Demand" column, table_id file_0_view_0
- $s_i$: "Initial Inventory" column, table_id file_0_view_0
- $x_i$: Decision variable for each $i \in I$
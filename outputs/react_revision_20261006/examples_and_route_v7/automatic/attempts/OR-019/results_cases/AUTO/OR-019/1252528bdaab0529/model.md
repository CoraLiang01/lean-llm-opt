##### Mathematical Model

Let $I$ be the set of products with 'Product Name' starting with '27in' (from table_id file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (column 'Revenue')
- $d_i$: Demand for product $i$ (column 'Demand')
- $s_i$: Initial Inventory for product $i$ (column 'Initial Inventory')

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. Demand fulfillment:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. Inventory limit:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 ('Product Name' with prefix '27in')
- $r_i$: 'Revenue' column, table_id file_0_view_0, for each $i \in I$
- $d_i$: 'Demand' column, table_id file_0_view_0, for each $i \in I$
- $s_i$: 'Initial Inventory' column, table_id file_0_view_0, for each $i \in I$
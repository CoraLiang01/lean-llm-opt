#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : set of all products classified as ‘27in’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$ : revenue per unit of product $i$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$ : total demand for product $i$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$ : initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

**Decision Variables:**
- $x_i$ : number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i,\, d_i\}, \quad \forall i \in I
   \]
   (Each fulfilled quantity cannot exceed either the initial inventory or the demand for that product.)

2. Variable Domain:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All rows in table_id file_0_view_0 where ‘Product Name’ contains or starts with ‘27in’ (column: ‘Product Name’).
- **Parameter $A_i$**: ‘Revenue’ column, table_id file_0_view_0.
- **Parameter $d_i$**: ‘Demand’ column, table_id file_0_view_0.
- **Parameter $I_i$**: ‘Initial Inventory’ column, table_id file_0_view_0.
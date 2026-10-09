#### Abstract Mathematical Model

**Index Set:**
- $I$ : set of all products with Product Name starting with ‘S700_’ (from table_id: file_0_view_0, column: Product Name)

**Parameters:**
- $A_i$ : revenue per unit of product $i \in I$ (from table_id: file_0_view_0, column: Revenue)
- $d_i$ : total demand for product $i \in I$ (from table_id: file_0_view_0, column: Demand)
- $I_i$ : initial inventory for product $i \in I$ (from table_id: file_0_view_0, column: Initial Inventory)

**Decision Variables:**
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id: file_0_view_0 where column ‘Product Name’ starts with ‘S700_’.
- **Parameter $A_i$:** table_id: file_0_view_0, column: Revenue.
- **Parameter $d_i$:** table_id: file_0_view_0, column: Demand.
- **Parameter $I_i$:** table_id: file_0_view_0, column: Initial Inventory.
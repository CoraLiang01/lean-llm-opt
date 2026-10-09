#### Symbolic Mathematical Model

**Index Set:**
- $I$ : set of all products classified under ‘Books’ (from table_id: file_0_view_0, column: Product_Name)

**Parameters:**
- $A_i$ : revenue per unit of product $i$ (from table_id: file_0_view_0, column: Revenue)
- $d_i$ : deterministic demand for product $i$ (from table_id: file_0_view_0, column: Demand)
- $I_i$ : initial inventory for product $i$ (from table_id: file_0_view_0, column: Initial Inventory)

**Decision Variables:**
- $x_i$ : number of units of product $i$ to fulfill, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Index set $I$ is defined by all records in table_id: file_0_view_0, column: Product_Name, filtered to products classified under ‘Books’.
- Parameter $A_i$ is mapped from table_id: file_0_view_0, column: Revenue.
- Parameter $d_i$ is mapped from table_id: file_0_view_0, column: Demand.
- Parameter $I_i$ is mapped from table_id: file_0_view_0, column: Initial Inventory.
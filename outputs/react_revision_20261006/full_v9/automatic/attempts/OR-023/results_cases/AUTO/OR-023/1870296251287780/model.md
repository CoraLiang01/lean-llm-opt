#### Mathematical Optimization Model

**Index Set:**
- $I$ : Set of all products with Product_Reference starting with "ELE-S" (from SalesStoreoverview.csv).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$ : Demand for product $i \in I$ (from column "Demand", table_id: file_0_view_0).
- $I_i$ : Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id: file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

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
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]

---

**Data Mapping:**

- Index set $I$ is defined by all rows in SalesStoreoverview.csv (table_id: file_0_view_0) where "Product_Reference" starts with "ELE-S".
- Parameter $A_i$ is mapped from column "Revenue" in table_id: file_0_view_0.
- Parameter $d_i$ is mapped from column "Demand" in table_id: file_0_view_0.
- Parameter $I_i$ is mapped from column "Initial Inventory" in table_id: file_0_view_0.
- Decision variable $x_i$ is defined for each $i \in I$.

No additional constraints or bounds are imposed beyond those specified above.
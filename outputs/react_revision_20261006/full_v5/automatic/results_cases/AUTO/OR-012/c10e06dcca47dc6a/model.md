#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products (from "Product Name" in table_id: file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from "Revenue", table_id: file_0_view_0).
- $d_i$: Demand for product $i$ (from "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from "Initial Inventory", table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- **Index Set $I$:** All "Product Name" entries in table_id: file_0_view_0, column "Product Name".
- **Parameter $A_i$:** "Revenue" column in table_id: file_0_view_0.
- **Parameter $d_i$:** "Demand" column in table_id: file_0_view_0.
- **Parameter $I_i$:** "Initial Inventory" column in table_id: file_0_view_0.
- **Variable $x_i$:** Decision variable for each $i \in I$.

**Source Table:**  
table_id: file_0_view_0  
Columns: "Product Name", "Revenue", "Demand", "Initial Inventory"
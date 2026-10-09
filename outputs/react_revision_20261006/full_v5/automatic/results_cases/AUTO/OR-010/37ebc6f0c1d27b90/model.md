#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products (from "Product Name" in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from "Revenue", file_0_view_0).
- $d_i$: Demand for product $i$ during the sales cycle (from "Demand", file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from "Initial Inventory", file_0_view_0).

**Decision Variables:**
- $x_i$: Number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (MobileSalesDataset.csv)
    - Index set $I$: "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"
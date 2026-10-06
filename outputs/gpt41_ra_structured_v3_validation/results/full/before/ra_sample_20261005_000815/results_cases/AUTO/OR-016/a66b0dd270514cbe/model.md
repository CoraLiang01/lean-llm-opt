#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all merchandise categories (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for category $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Total demand for category $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for category $i$ (from column "Initial Inventory", table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units to fulfill for category $i$.
  - Domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
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
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (RetailSalesDataset.csv)
    - Index set $I$: All unique "Product Name" values
    - Parameter $A_i$: "Revenue" column
    - Parameter $d_i$: "Demand" column
    - Parameter $I_i$: "Initial Inventory" column
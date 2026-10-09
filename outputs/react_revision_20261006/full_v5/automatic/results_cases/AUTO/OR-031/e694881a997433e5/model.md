#### Mathematical Optimization Model

**Index Set:**
- $I$: set of all dairy products, indexed by $i$ (from column "Full_Product_Name" in table_id file_0_view_0).

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: deterministic demand for product $i$ (from column "Demand", table_id file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory", table_id file_0_view_0).

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory limit:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand limit:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (DairyGoodsSalesDataset.csv)
    - Index set $I$: Full_Product_Name
    - Parameter $A_i$: Revenue
    - Parameter $d_i$: Demand
    - Parameter $I_i$: Initial Inventory
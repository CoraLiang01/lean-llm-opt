#### Mathematical Optimization Model

**Index Set:**
- $I$: set of all dairy products, indexed by $i$ (from all records in table_id = file_0_view_0, column "Full_Product_Name")

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from table_id = file_0_view_0, column "Revenue")
- $d_i$: deterministic demand for product $i$ (from table_id = file_0_view_0, column "Demand")
- $I_i$: initial inventory for product $i$ (from table_id = file_0_view_0, column "Initial Inventory")

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$, with $x_i \geq 0$ and integer

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
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping:**

- Index set $I$ and all parameters $A_i$, $d_i$, $I_i$ are mapped from table_id = file_0_view_0 in "DairyGoodsSalesDataset.csv" using columns:
    - "Full_Product_Name" (for $i$)
    - "Revenue" (for $A_i$)
    - "Demand" (for $d_i$)
    - "Initial Inventory" (for $I_i$)
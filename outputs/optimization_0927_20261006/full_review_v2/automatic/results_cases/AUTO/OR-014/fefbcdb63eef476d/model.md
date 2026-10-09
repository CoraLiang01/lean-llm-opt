#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: set of all pizza types (from column "Product Name" in table_id: file_0_view_0).

**Parameters:**
- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: total demand for pizza type $i$ (from column "Demand" in table_id: file_0_view_0).
- $I_i$: initial inventory available for pizza type $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraints:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- All index sets and parameters are drawn from table_id: file_0_view_0 (PizzaSalesDataset.csv), using columns:
    - "Product Name" $\rightarrow$ $I$
    - "Revenue" $\rightarrow$ $A_i$
    - "Demand" $\rightarrow$ $d_i$
    - "Initial Inventory" $\rightarrow$ $I_i$
- No additional filters were applied; all records in the table are included.
#### Abstract Mathematical Optimization Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all pizza types, as identified by the column "Product Name" in table_id file_0_view_0.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of pizza type $i$ ("Revenue", file_0_view_0)
- $d_i$: Total demand for pizza type $i$ ("Demand", file_0_view_0)
- $I_i$: Initial inventory available for pizza type $i$ ("Initial Inventory", file_0_view_0)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer)

**Objective:**  
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

**Data Mapping:**  
- Index set $\mathcal{I}$: All unique values in "Product Name" from table_id file_0_view_0 (PizzaSalesDataset.csv)
- Parameter $A_i$: "Revenue" column, table_id file_0_view_0
- Parameter $d_i$: "Demand" column, table_id file_0_view_0
- Parameter $I_i$: "Initial Inventory" column, table_id file_0_view_0

No literal data values or record counts are included; all identifiers and columns are preserved as in the source.
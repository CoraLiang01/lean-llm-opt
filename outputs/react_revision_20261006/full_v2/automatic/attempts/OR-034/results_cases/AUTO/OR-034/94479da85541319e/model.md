#### Mathematical Optimization Model

**Index Set:**
- $i \in \mathcal{P}$: Set of all baked goods (Product Name) in the dataset.

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Demand for baked good $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for baked good $i$ (from column "Initial Inventory", table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in \mathcal{P}$.

**Objective:**
\[
\max \sum_{i \in \mathcal{P}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{P}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{P}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{P}
   \]

---

#### Data Mapping

- $\mathcal{P}$: All "Product Name" entries in table_id: file_0_view_0 (Frenchbakerydailysales.csv).
- $A_i$: "Revenue" column, table_id: file_0_view_0.
- $d_i$: "Demand" column, table_id: file_0_view_0.
- $I_i$: "Initial Inventory" column, table_id: file_0_view_0.
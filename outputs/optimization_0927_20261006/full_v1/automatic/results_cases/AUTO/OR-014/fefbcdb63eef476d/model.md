#### Abstract Mathematical Optimization Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all pizza types, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit of pizza type $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Total demand for pizza type $i$ over the sales horizon (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory available for pizza type $i$ (from column "Initial Inventory", table_id: file_0_view_0)

**Decision Variables:**  
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer), for all $i \in \mathcal{I}$

**Objective:**  
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   \]

---

**Data Mapping:**  
- Table: PizzaSalesDataset.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $I_i$: column "Initial Inventory"
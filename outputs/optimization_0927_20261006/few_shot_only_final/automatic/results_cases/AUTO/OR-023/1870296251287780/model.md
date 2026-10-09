**Sets:**  
- $I$: Index set of all products with classification ‘ELE-S’ (as defined by the user).

**Parameters:**  
- $A_i$: Revenue per unit of product $i$, from column ‘Revenue’ in table_id.
- $d_i$: Total deterministic demand for product $i$, from column ‘Demand’ in table_id.
- $I_i$: Initial inventory for product $i$, from column ‘Initial Inventory’ in table_id.

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**  
- Table: file_0_view_0 (from SalesStoreoverview.csv)
    - Product index set $I$: All rows where the product is classified as ‘ELE-S’ (user-defined).
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column

---

**Abstract Model Summary:**  
\[
\begin{align*}
\max_{x_i} \quad & \sum_{i \in I} A_i x_i \\
\text{s.t.} \quad & x_i \leq I_i, \quad \forall i \in I \\
                  & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
\]
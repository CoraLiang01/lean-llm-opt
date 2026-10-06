**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products with $id\_number$ prefix ‘id999’ (from file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from Revenue column, file_0_view_0).
- $d_i$: Demand for product $i$ during the sales horizon (from Demand column, file_0_view_0).
- $s_i$: Initial inventory of product $i$ (from Initial Inventory column, file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   $\quad x_i \leq s_i, \quad \forall i \in I$

2. **Demand Constraint:**  
   $\quad x_i \leq d_i, \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I$

---

**Data Mapping**

- $I$: All rows in file_0_view_0 where $id\_number$ starts with ‘id999’.
- $r_i$: file_0_view_0, column Revenue, for product $i$.
- $d_i$: file_0_view_0, column Demand, for product $i$.
- $s_i$: file_0_view_0, column Initial Inventory, for product $i$.
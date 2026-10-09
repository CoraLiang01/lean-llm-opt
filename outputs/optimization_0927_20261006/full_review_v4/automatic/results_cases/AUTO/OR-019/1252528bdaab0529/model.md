#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified under ‘27in’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Index set $I$, and parameters $A_i$, $d_i$, $I_i$ are defined using:
    - table_id: file_0_view_0
    - columns: ‘Product Name’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’
    - filter: rows where ‘Product Name’ has prefix ‘27in’ (as returned by CSVQA)
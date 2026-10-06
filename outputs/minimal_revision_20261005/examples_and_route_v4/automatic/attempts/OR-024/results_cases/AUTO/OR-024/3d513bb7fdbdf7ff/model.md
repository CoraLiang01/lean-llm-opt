**Mathematical Optimization Model**

---

**Index Sets:**

- $I$: Set of products with names beginning 'S700_' (from column ‘Product Name’ in file_0_view_0).

---

**Parameters:**

- $r_i$: Revenue per unit of product $i$ (from ‘Revenue’ in file_0_view_0, indexed by ‘Product Name’).
- $d_i$: Demand for product $i$ (from ‘Demand’ in file_0_view_0, indexed by ‘Product Name’).
- $s_i$: Initial inventory of product $i$ (from ‘Initial Inventory’ in file_0_view_0, indexed by ‘Product Name’).

---

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

---

**Objective:**

\[
\max \sum_{i \in I} r_i x_i
\]

---

**Constraints:**

1. **Demand fulfillment:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

2. **Inventory availability:**
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in file_0_view_0 where ‘Product Name’ starts with ‘S700_’.
- $r_i$: file_0_view_0, column ‘Revenue’, key ‘Product Name’.
- $d_i$: file_0_view_0, column ‘Demand’, key ‘Product Name’.
- $s_i$: file_0_view_0, column ‘Initial Inventory’, key ‘Product Name’.
**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products classified under ‘27in’ (from all rows in Salesorders.csv with "Product Name" starting with "27in").

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $d_i$: Demand for product $i$ (from Demand column).
- $s_i$: Initial inventory of product $i$ (from Initial Inventory column).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

2. **Demand constraint:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

3. **Nonnegativity and integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

**Data Mapping**

- $I$: All records in `Salesorders.csv` where `Product Name` starts with "27in" (see table_id: file_0_view_0).
- $r_i$: `Revenue` column, table_id: file_0_view_0, for each $i \in I$.
- $d_i$: `Demand` column, table_id: file_0_view_0, for each $i \in I$.
- $s_i$: `Initial Inventory` column, table_id: file_0_view_0, for each $i \in I$.

**Variable:**
- $x_i$: Number of units of product $i$ to fulfill, for each $i \in I$.
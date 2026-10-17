#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products $i$ such that the value in column `Product Name` of table `file_0_view_0` contains the substring "Fashion".

**Parameters:**
- $A_i$: Revenue per unit of product $i$, from column `Revenue` in table `file_0_view_0`.
- $d_i$: Deterministic demand for product $i$, from column `Demand` in table `file_0_view_0`.
- $I_i$: Initial inventory for product $i$, from column `Initial Inventory` in table `file_0_view_0$.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, for $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

**Data Mapping:**

- **Source Table:** `file_0_view_0` (from `SupermarketSales.csv`)
- **Index Set $I$:** All rows where `Product Name` contains the substring "Fashion"
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`
- **Variable $x_i$:** Decision variable for each $i \in I$ (Fashion products only)

---

**Notes:**
- All rows and columns are preserved in source order.
- Only products with "Fashion" in the `Product Name` are included in the index set $I$ as per the query.
- No additional constraints or selection rules are imposed beyond those stated in the query.
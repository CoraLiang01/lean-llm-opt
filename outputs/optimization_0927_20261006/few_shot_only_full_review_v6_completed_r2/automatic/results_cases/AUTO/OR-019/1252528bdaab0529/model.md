**Abstract Mathematical Model**

---

**Index Sets**

- Let $\mathcal{I}$ be the set of all products $i$ such that the value in column `Product Name` contains the substring `'27in'` in table `file_0_view_0`.

---

**Parameters**

For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column `Revenue` in table `file_0_view_0`)
- $d_i$: Demand for product $i$ (from column `Demand` in table `file_0_view_0`)
- $I_i$: Initial inventory of product $i$ (from column `Initial Inventory` in table `file_0_view_0`)

---

**Decision Variables**

For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

---

**Objective**

\[
\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

---

**Constraints**

For all $i \in \mathcal{I}$:
1. **Demand Constraint:**  
   \[
   x_i \leq d_i
   \]
2. **Inventory Constraint:**  
   \[
   x_i \leq I_i
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

**Data Mapping**

- **Table:** `file_0_view_0` (from `SalesDataAnalysis.csv`)
- **Index Set Selection:** $\mathcal{I} = \{i : \text{`Product Name`} \text{ contains } '27in'\}$
- **Parameters:**
    - $A_i$ from column `Revenue`
    - $d_i$ from column `Demand`
    - $I_i$ from column `Initial Inventory`

---

**Summary:**  
This model maximizes total revenue from all products whose names contain '27in', subject to demand and inventory constraints, using the exact data columns and selection criteria specified above.
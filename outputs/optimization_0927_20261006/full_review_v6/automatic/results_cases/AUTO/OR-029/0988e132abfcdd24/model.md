#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified under ‘FAUX’ (from Product Name column).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from Revenue column).
- $d_i$: Total demand for product $i \in I$ (from Demand column).
- $I_i$: Initial inventory for product $i \in I$ (from Initial Inventory column).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (ZARASales.csv)
- **Columns Used:**
  - Product Name (filtered by prefix ‘FAUX’)
  - Revenue
  - Demand
  - Initial Inventory

- **Filter Applied:** Product Name starts with ‘FAUX’ (as validated and returned by CSVQA).
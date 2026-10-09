#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘Books’ (from the filtered dataset).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory available for product $i \in I$ (from column ‘Initial Inventory’).

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

- **Table:** file_0_view_0 (from DifferentStoreSales.csv)
- **Columns Used:**
  - Product_Name (filtered by prefix ‘Books’)
  - Revenue
  - Initial Inventory
  - Demand

- **Filter Applied:** Only rows where Product_Name starts with ‘Books’ are included.
#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all car models classified as ‘FDK57’ (indexed by $i$).

**Parameters:**
- $A_i$ : Revenue per unit for car model $i$ (from column ‘Revenue’).
- $d_i$ : Total deterministic demand for car model $i$ (from column ‘Demand’).
- $I_i$ : Initial inventory for car model $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of car model $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

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

- **Table:** file_0_view_0 (from BigMartSales.csv)
- **Columns Used:**
  - Product Name (filtered: exact match ‘FDK57’)
  - Revenue (parameter $A_i$)
  - Demand (parameter $d_i$)
  - Initial Inventory (parameter $I_i$)
- **Filter Applied:** Product Name = ‘FDK57’ (as returned by CSVQA; all records with this exact value are included)
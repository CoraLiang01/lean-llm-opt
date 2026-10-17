#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : Set of all car models classified as ‘FDK57’ (indexed by $i$).

**Parameters:**
- $A_i$ : Revenue per unit for car model $i$ (from column ‘Revenue’).
- $I_i$ : Initial inventory for car model $i$ (from column ‘Initial Inventory’).
- $d_i$ : Demand for car model $i$ (from column ‘Demand’).

**Decision Variables:**
- $x_i$ : Number of units of car model $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
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
  - Revenue
  - Initial Inventory
  - Demand

All records returned by the filter Product Name = ‘FDK57’ are included as the index set $I$ and parameterized by their respective column values.
#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products with identifiers starting with ‘S700_’.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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

- **Table:** file_0_view_0 (SampleSalesData.csv)
- **Columns:**
  - Product Name (filtered: prefix = ‘S700_’)
  - Revenue
  - Demand
  - Initial Inventory

All parameters and index sets are defined using the records returned by this filter. No additional filtering or data sources are used.
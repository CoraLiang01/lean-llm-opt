#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified under ‘Organ’ (as defined by the data source).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

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

- **Table:** SupermartGrocerySales-RetailAnalyticsDataset.csv
- **Index Set:** $I$ = all records where [Sub Category] has prefix "Organic"
- **Parameters:**
    - $A_i$: [Revenue] column, for each $i \in I$
    - $d_i$: [Demand] column, for each $i \in I$
    - $I_i$: [Initial Inventory] column, for each $i \in I$
- **Decision Variables:** $x_i$ for each $i \in I$
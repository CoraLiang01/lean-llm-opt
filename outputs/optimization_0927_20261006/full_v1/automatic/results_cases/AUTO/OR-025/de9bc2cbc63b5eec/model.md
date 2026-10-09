#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all ‘TABLET’ smartphone models (from the data source).

**Parameters:**
- $A_i$: Revenue per unit for model $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for model $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for model $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of model $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

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

- **Source Table:** SmartphoneRetailOutletSalesData.csv
- **Index Set:** All rows where ‘Product Name’ has prefix ‘TABLET’
- **Parameters:**
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $I_i$: column ‘Initial Inventory’
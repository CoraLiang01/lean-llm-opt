#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all ‘TABLET’ smartphone models (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for model $i$ (from column ‘Revenue’).
- $d_i$: Total demand for model $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for model $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of model $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

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
- **Index Set $I$:** All rows where [Product Name] has prefix ‘TABLET_’ (i.e., [Product Name] starts with ‘TABLET_’)
- **Parameter $A_i$:** [Revenue] column, for each $i \in I$
- **Parameter $d_i$:** [Demand] column, for each $i \in I$
- **Parameter $I_i$:** [Initial Inventory] column, for each $i \in I$
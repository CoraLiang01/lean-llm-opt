#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products classified under ‘ZZ’ (indexed by $i$).

**Parameters:**
- $A_i$ : Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i$ (from column ‘Demand’).
- $I_i$ : Initial inventory of product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

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

- **Table:** RetailStoreSalesTransactions(ScannerData).csv
- **Index Set $I$:** All rows where column ‘SKU’ has prefix ‘ZZ’
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’
#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified under ‘ZZ’ (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory constraint:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from RetailStoreSalesTransactions(ScannerData).csv)
- **Index Set $I$:** All records where column ‘SKU’ has prefix ‘ZZ’ (CSVQA predicate: SKU prefix ‘ZZ’)
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’
#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified under ‘ZZ’ (indexed by $i$).

**Parameters:**
- $A_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:**   $x_i \leq I_i \quad \forall i \in I$
3. **Non-negativity and integrality:** $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id: file_0_view_0 where column ‘SKU’ has prefix ‘ZZ’.
- **Parameter $A_i$:** Column ‘Revenue’ in table_id: file_0_view_0.
- **Parameter $d_i$:** Column ‘Demand’ in table_id: file_0_view_0.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in table_id: file_0_view_0.
- **Decision Variable $x_i$:** Number of units fulfilled for each $i \in I$.

All data is sourced from table_id: file_0_view_0, columns: ‘SKU’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’ in RetailStoreSalesTransactions(ScannerData).csv, filtered to rows where ‘SKU’ starts with ‘ZZ’.
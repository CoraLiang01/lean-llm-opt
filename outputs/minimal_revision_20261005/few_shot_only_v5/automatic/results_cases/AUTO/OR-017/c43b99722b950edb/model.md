#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all product categories classified under 'ZZ'.

**Parameters:**

- $r_i$ : Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$ : Deterministic total demand for product $i \in I$ (from column 'Demand').
- $s_i$ : Initial inventory for product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**

1. **Demand fulfillment:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Inventory availability:**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv`
- **Columns Used:**
  - `SKU` (for index set $I$)
  - `Revenue` (for $r_i$)
  - `Demand` (for $d_i$)
  - `Initial Inventory` (for $s_i$)
#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all product categories with SKU codes classified under ‘ZZ’.

**Parameters:**

- $a_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Deterministic total demand for product $i \in I$ (from column ‘Demand’).
- $s_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $\forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**

1. **Demand fulfillment:**
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
2. **Inventory limit:**
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `RetailStoreSalesTransactions(ScannerData).csv` (`file_0_view_0`)
- **Index Set $I$:** All rows where the SKU code (column `SKU`) is classified under ‘ZZ’ (i.e., SKU starts with or contains 'ZZ').
- **Parameter $a_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`

---

**Note:** All data for $I$, $a_i$, $d_i$, $s_i$ are to be taken directly from the specified columns and table, using the full set of ‘ZZ’ SKUs as defined in the dataset.
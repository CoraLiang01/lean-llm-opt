#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all products with `Product_Reference` beginning with "ELE-S" (i.e., all 'ELE-S' products) from table `file_0_view_0`.

**Parameters:**

- $a_i$ : Revenue per unit of product $i \in I$ (`Revenue`, table `file_0_view_0`).
- $d_i$ : Demand for product $i \in I$ (`Demand`, table `file_0_view_0`).
- $s_i$ : Initial inventory for product $i \in I$ (`Initial Inventory`, table `file_0_view_0`).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

\[
\max \quad \sum_{i \in I} a_i x_i
\]

**Constraints:**

1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Inventory constraint:**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SalesStoreoverview.csv`)
- **Index Set:** $I$ = all rows where `Product_Reference` starts with "ELE-S"
- **Parameter $a_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`
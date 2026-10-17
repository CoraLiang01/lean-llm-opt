#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products such that the value in column `Product_Reference` begins with "ELE-S" (i.e., all products classified under ‘ELE-S’).

**Parameters:**

- $r_i$ : Revenue per unit of product $i \in I$ (from column `Revenue`).
- $d_i$ : Demand for product $i \in I$ (from column `Demand`).
- $s_i$ : Initial inventory for product $i \in I$ (from column `Initial Inventory`).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
2. **Inventory constraint:**
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `SalesStoreoverview.csv`)
- **Index Set $I$:** All rows where `Product_Reference` starts with "ELE-S"
- **Parameter $r_i$:** Column `Revenue` in `file_0_view_0`
- **Parameter $d_i$:** Column `Demand` in `file_0_view_0`
- **Parameter $s_i$:** Column `Initial Inventory` in `file_0_view_0`
- **Decision Variable $x_i$:** Defined for each $i \in I$ as above

---

**Note:** No literal data or record counts are included. All selection and mapping are defined symbolically as per the query and the returned table.
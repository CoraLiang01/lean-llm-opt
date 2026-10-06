#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of baked goods, indexed by $i$ (from column `Product Name` in `file_0_view_0`).

**Parameters:**
- $r_i$: Revenue per unit of baked good $i$ (`Revenue`, table_id: file_0_view_0).
- $d_i$: Demand for baked good $i$ (`Demand`, table_id: file_0_view_0).
- $s_i$: Initial inventory of baked good $i$ (`Initial Inventory`, table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory and demand fulfillment:
   $$
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   $$

2. Integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$: All rows in `file_0_view_0` (`Product Name`)
- $r_i$: `Revenue` column, table_id: file_0_view_0, key: `Product Name`
- $d_i$: `Demand` column, table_id: file_0_view_0, key: `Product Name`
- $s_i$: `Initial Inventory` column, table_id: file_0_view_0, key: `Product Name`
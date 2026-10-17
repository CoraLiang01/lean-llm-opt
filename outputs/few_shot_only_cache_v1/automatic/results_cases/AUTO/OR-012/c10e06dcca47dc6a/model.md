#### Abstract Mathematical Model

**Index Sets:**
- $P$: Set of all products (indexed by $i$), corresponding to all rows in table_id = file_0_view_0, column "Product Name".

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from table_id = file_0_view_0, column "Revenue").
- $d_i$: Deterministic demand for product $i$ over the sales horizon (from table_id = file_0_view_0, column "Demand").
- $I_i$: Initial inventory available for product $i$ (from table_id = file_0_view_0, column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in P$.

**Objective:**
\[
\max \sum_{i \in P} r_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i, \quad \forall i \in P
   \]
2. **Inventory constraint:**
   \[
   x_i \leq I_i, \quad \forall i \in P
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

---

#### Data Mapping

- **Index Set $P$:** All rows in table_id = file_0_view_0, column "Product Name".
- **Parameter $r_i$:** table_id = file_0_view_0, column "Revenue".
- **Parameter $d_i$:** table_id = file_0_view_0, column "Demand".
- **Parameter $I_i$:** table_id = file_0_view_0, column "Initial Inventory".

All data is mapped directly from the columns of table_id = file_0_view_0 as returned by CSVQA. No rows are filtered or omitted.
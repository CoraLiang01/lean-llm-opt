#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Total deterministic demand for product $i$ over the sales horizon (from column "Demand").
- $I_i$: Initial inventory available for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory constraint:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All product names from table_id: `file_0_view_0`, column: "Product Name".
- **Parameter $A_i$:** Revenue per unit from table_id: `file_0_view_0`, column: "Revenue".
- **Parameter $d_i$:** Demand from table_id: `file_0_view_0`, column: "Demand".
- **Parameter $I_i$:** Initial Inventory from table_id: `file_0_view_0`, column: "Initial Inventory".
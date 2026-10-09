#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products with Product_Reference starting with "ELE-S".

**Parameters:**
- $a_i$: Revenue per unit of product $i$, from column "Revenue" in table_id file_0_view_0.
- $d_i$: Demand for product $i$, from column "Demand" in table_id file_0_view_0.
- $s_i$: Initial inventory for product $i$, from column "Initial Inventory" in table_id file_0_view_0.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All records in table_id file_0_view_0 with Product_Reference prefix "ELE-S".
- **Parameter $a_i$**: file_0_view_0, column "Revenue".
- **Parameter $d_i$**: file_0_view_0, column "Demand".
- **Parameter $s_i$**: file_0_view_0, column "Initial Inventory".
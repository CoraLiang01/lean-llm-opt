#### Index Sets

- $I$: Set of all products with SKU prefix ‘ZZ’ (from table_id: file_0_view_0, column: SKU).

#### Parameters

- $r_i$: Revenue per unit of product $i$ (from file_0_view_0, column: Revenue).
- $d_i$: Total deterministic demand for product $i$ (from file_0_view_0, column: Demand).
- $s_i$: Initial inventory for product $i$ (from file_0_view_0, column: Initial Inventory).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

#### Objective

\[
\max \quad \sum_{i \in I} r_i \, x_i
\]

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All SKUs in table_id: file_0_view_0, column: SKU, with prefix ‘ZZ’.
- **Parameter $r_i$:** file_0_view_0, column: Revenue.
- **Parameter $d_i$:** file_0_view_0, column: Demand.
- **Parameter $s_i$:** file_0_view_0, column: Initial Inventory.
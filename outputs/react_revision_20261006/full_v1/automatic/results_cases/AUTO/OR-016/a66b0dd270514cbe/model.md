#### Sets
Let $\mathcal{I}$ be the set of all products, indexed by $i$.

#### Parameters
- $r_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in \mathcal{I}$

#### Objective
\[
\max \sum_{i \in \mathcal{I}} r_i x_i
\]

#### Constraints
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in \mathcal{I}
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- Set $\mathcal{I}$: All "Product Name" entries in table_id: file_0_view_0, column "Product Name"
- Parameter $r_i$: table_id: file_0_view_0, column "Revenue"
- Parameter $d_i$: table_id: file_0_view_0, column "Demand"
- Parameter $s_i$: table_id: file_0_view_0, column "Initial Inventory"
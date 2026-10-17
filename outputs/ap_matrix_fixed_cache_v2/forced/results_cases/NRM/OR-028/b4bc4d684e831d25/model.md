#### Index Sets

- $I$: set of all products, indexed by $i$.

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0)
- $d_i$: total demand for product $i$ over the sales horizon (from column "Demand" in table_id file_0_view_0)
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory" in table_id file_0_view_0)

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. Inventory constraints:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]

2. Demand constraints:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id file_0_view_0, columns "Revenue", "Demand", and "Initial Inventory", respectively, for all products (no filter; all records used).
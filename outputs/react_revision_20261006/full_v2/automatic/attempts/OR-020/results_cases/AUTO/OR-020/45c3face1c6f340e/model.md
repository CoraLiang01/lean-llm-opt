#### Sets
- $I$: set of all products, indexed by $i$.

#### Parameters
- $r_i$: revenue per unit of product $i$ (from column "Revenue").
- $d_i$: total demand for product $i$ (from column "Demand").
- $s_i$: initial inventory of product $i$ (from column "Initial Inventory").

#### Decision Variables
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} r_i x_i
\]

#### Constraints
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: file_0_view_0 (from SalesDatainBusinesses.csv)
    - Product set $I$: all unique values in column "Product Name"
    - Revenue parameter $r_i$: column "Revenue"
    - Demand parameter $d_i$: column "Demand"
    - Initial inventory parameter $s_i$: column "Initial Inventory"
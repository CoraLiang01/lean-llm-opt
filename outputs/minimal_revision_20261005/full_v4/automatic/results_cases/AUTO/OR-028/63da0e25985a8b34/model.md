#### Index Sets

- $I$: set of all products, indexed by $i$.

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: total demand for product $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i \, x_i
\]

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: file_0_view_0
    - Product Name: index set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$
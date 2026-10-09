#### Index Sets

- $I$: Set of all products with "27in" in the "Product Name" column of table_id = file_0_view_0.

#### Parameters

- $A_i$: Revenue per unit of product $i$, from column "Revenue" in table_id = file_0_view_0.
- $d_i$: Demand for product $i$, from column "Demand" in table_id = file_0_view_0.
- $I_i$: Initial inventory for product $i$, from column "Initial Inventory" in table_id = file_0_view_0.

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

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

- **table_id:** file_0_view_0
- **Product Name:** Used to define set $I$ (all products containing "27in")
- **Revenue:** Parameter $A_i$
- **Demand:** Parameter $d_i$
- **Initial Inventory:** Parameter $I_i$
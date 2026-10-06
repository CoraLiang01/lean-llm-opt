#### Index Sets

- $I$: Set of all products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total demand for product $i$.  
  (From column "Demand" in table_id: file_0_view_0)
- $s_i$: Initial inventory for product $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i x_i
\]

#### Constraints

1. **Demand fulfillment constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Inventory constraint:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **table_id:** file_0_view_0
- **Product Name:** Used to define index set $I$.
- **Revenue:** Parameter $A_i$.
- **Demand:** Parameter $d_i$.
- **Initial Inventory:** Parameter $s_i$.
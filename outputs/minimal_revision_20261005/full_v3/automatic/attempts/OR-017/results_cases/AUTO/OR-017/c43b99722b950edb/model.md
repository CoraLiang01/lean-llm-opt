#### Index Sets

- $I$: Set of all products classified under ‘ZZ’, indexed by $i$.

#### Parameters

- $r_i$: Revenue per unit of product $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total demand for product $i$ over the sales horizon.  
  (From column "Demand" in table_id: file_0_view_0)
- $s_i$: Initial inventory of product $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \quad \sum_{i \in I} r_i x_i
\]

#### Constraints

1. **Inventory Constraints:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

2. **Demand Constraints:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **SKU:** Index set $I$
    - **Revenue:** $r_i$
    - **Demand:** $d_i$
    - **Initial Inventory:** $s_i$
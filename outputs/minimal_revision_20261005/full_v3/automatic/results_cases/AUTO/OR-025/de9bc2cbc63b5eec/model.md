#### Index Sets

- $I$: Set of all ‘TABLET’ smartphone models, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for model $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total demand for model $i$.  
  (From column "Demand" in table_id: file_0_view_0)
- $I_i$: Initial inventory for model $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of model $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \, x_i
$$

#### Constraints

1. **Inventory Constraint:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraint:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Nonnegativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name:** Index set $I$
    - **Revenue:** Parameter $A_i$
    - **Demand:** Parameter $d_i$
    - **Initial Inventory:** Parameter $I_i$
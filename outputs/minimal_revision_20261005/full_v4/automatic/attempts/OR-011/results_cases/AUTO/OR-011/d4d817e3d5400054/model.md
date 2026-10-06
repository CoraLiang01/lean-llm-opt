#### Index Sets

- $I$: Set of all products with $id\_number$ prefix ‘id999’ (from table_id = file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Demand for product $i$ during the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **id_number:** Index set $I$ (products with $id\_number$ prefix ‘id999’)
    - **Revenue:** Parameter $A_i$
    - **Demand:** Parameter $d_i$
    - **Initial Inventory:** Parameter $I_i$
#### Index Sets

- $I$: Set of all products classified as ‘Fashion’, indexed by $i$.

#### Parameters

- $r_i$: Revenue per unit of product $i$.  
- $d_i$: Deterministic demand for product $i$ over the planning horizon.  
- $s_i$: Initial inventory available for product $i$.

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint:**  
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$

2. **Demand Constraint:**  
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- $I$: All rows in table_id = file_0_view_0 where [Product Name] is classified under ‘Fashion’ (see filter in CSVQA_DATA).
- $r_i$: [Revenue] column, table_id = file_0_view_0, indexed by [Product Name].
- $s_i$: [Initial Inventory] column, table_id = file_0_view_0, indexed by [Product Name].
- $d_i$: [Demand] column, table_id = file_0_view_0, indexed by [Product Name].
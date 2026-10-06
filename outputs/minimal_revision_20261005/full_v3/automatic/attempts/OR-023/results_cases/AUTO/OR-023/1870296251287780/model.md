#### Index Sets

- $I$: Set of all products with ‘Product_Reference’ starting with ‘ELE-S’ (from table_id: file_0_view_0).

#### Parameters

- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Demand for product $i$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint**  
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$

2. **Demand Constraint**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Variable Domain**  
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$**: All rows in table_id: file_0_view_0 where ‘Product_Reference’ starts with ‘ELE-S’ (column: Product_Reference).
- **Parameter $r_i$**: table_id: file_0_view_0, column: Revenue.
- **Parameter $d_i$**: table_id: file_0_view_0, column: Demand.
- **Parameter $s_i$**: table_id: file_0_view_0, column: Initial Inventory.
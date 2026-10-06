#### Index Sets

- $I$: Set of all products with Product_Reference starting with ‘ELE-S’ (from table_id = file_0_view_0, column Product_Reference).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from file_0_view_0, column Revenue).
- $d_i$: Demand for product $i$ (from file_0_view_0, column Demand).
- $I_i$: Initial inventory for product $i$ (from file_0_view_0, column Initial Inventory).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$**: All records in table_id = file_0_view_0, column Product_Reference, where Product_Reference starts with ‘ELE-S’.
- **Parameter $A_i$**: file_0_view_0, column Revenue.
- **Parameter $d_i$**: file_0_view_0, column Demand.
- **Parameter $I_i$**: file_0_view_0, column Initial Inventory.
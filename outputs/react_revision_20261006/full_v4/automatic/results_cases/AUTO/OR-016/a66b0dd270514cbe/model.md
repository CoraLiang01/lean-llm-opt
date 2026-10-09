#### Sets
Let $I$ be the set of all products, indexed by $i$.

#### Parameters
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from table_id: file_0_view_0, column: Revenue)
- $d_i$: Demand for product $i$ (from table_id: file_0_view_0, column: Demand)
- $I_i$: Initial Inventory for product $i$ (from table_id: file_0_view_0, column: Initial Inventory)

#### Decision Variables
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints

1. Inventory constraints:
$$
x_i \leq I_i \quad \forall i \in I
$$

2. Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

---

#### Data Mapping

- Set $I$: All products in table_id: file_0_view_0, column: Product Name
- Parameter $A_i$: table_id: file_0_view_0, column: Revenue
- Parameter $d_i$: table_id: file_0_view_0, column: Demand
- Parameter $I_i$: table_id: file_0_view_0, column: Initial Inventory
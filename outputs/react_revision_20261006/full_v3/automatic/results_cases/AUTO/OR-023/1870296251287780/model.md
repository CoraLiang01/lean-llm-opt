#### Sets

Let $I$ be the set of all products with Product_Reference starting with "ELE-S" in table_id file_0_view_0.

#### Parameters

For each $i \in I$:
- $a_i$: Revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (from column "Initial Inventory", table_id file_0_view_0)

#### Decision Variables

For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective

Maximize total revenue:
$$
\max \sum_{i \in I} a_i x_i
$$

#### Constraints

Inventory and demand bounds:
$$
x_i \leq s_i, \quad \forall i \in I
$$
$$
x_i \leq d_i, \quad \forall i \in I
$$
$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
$$

#### Data Mapping

- Set $I$: All rows in table_id file_0_view_0 where "Product_Reference" has prefix "ELE-S"
- Parameter $a_i$: "Revenue" column, table_id file_0_view_0
- Parameter $d_i$: "Demand" column, table_id file_0_view_0
- Parameter $s_i$: "Initial Inventory" column, table_id file_0_view_0
- Variable $x_i$: Decision variable for each $i \in I$
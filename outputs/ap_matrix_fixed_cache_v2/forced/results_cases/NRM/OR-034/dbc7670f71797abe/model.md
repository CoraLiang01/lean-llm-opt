#### Index Sets

- $I$: set of all baked goods (indexed by $i$), as listed in column "Product Name" of table_id: file_0_view_0.

#### Parameters

- $A_i$: revenue per unit of baked good $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: total demand for baked good $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: initial inventory for baked good $i$ (from column "Initial Inventory", table_id: file_0_view_0)

#### Decision Variables

- $x_i$: quantity of baked good $i$ to fulfill, integer, $x_i \geq 0$, for all $i \in I$

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Bounds:
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

2. Integrality:
   $$
   x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

#### Data Mapping

- All sets and parameters are drawn from table_id: file_0_view_0 in "Frenchbakerydailysales.csv":
    - $I$: column "Product Name"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $I_i$: column "Initial Inventory"
#### Sets
- $I$: set of all products, indexed by $i$

#### Parameters
- $A_i$: revenue per unit of product $i$ (from table_id: file_0_view_0, column: Revenue)
- $d_i$: deterministic demand for product $i$ (from table_id: file_0_view_0, column: Demand)
- $I_i$: initial inventory for product $i$ (from table_id: file_0_view_0, column: Initial Inventory)

#### Decision Variables
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, $\forall i \in I$

#### Objective
$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints
1. Inventory constraint:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

#### Data Mapping
- Set $I$, parameters $A_i$, $d_i$, $I_i$ are defined from table_id: file_0_view_0, columns: Product Name, Revenue, Demand, Initial Inventory.
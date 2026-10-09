#### Index Sets
- $I$: set of all products (indexed by $i$), corresponding to all "Product Name" entries in table_id file_0_view_0.

#### Parameters
- $A_i$: revenue per unit of product $i$ ("Revenue" column, table_id file_0_view_0).
- $d_i$: deterministic demand for product $i$ ("Demand" column, table_id file_0_view_0).
- $I_i$: initial inventory for product $i$ ("Initial Inventory" column, table_id file_0_view_0).

#### Decision Variables
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
- Demand and inventory bounds:
  $$
  0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
  $$
- $x_i$ is continuous or integer as required by the application (not specified as integer in the query, so default to $x_i \geq 0$ and real-valued unless otherwise required).

#### Data Mapping

- All sets and parameters are mapped from table_id file_0_view_0, columns:
    - Product Name $\rightarrow$ $I$
    - Revenue $\rightarrow$ $A_i$
    - Demand $\rightarrow$ $d_i$
    - Initial Inventory $\rightarrow$ $I_i$
- The model uses all records returned by CSVQA from file_0_view_0, with no additional filtering.
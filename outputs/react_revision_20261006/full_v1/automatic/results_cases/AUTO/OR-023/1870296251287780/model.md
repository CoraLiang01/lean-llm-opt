#### Sets
- $I$: Set of all products with Product_Reference starting with "ELE-S" (from table_id file_0_view_0, column Product_Reference).

#### Parameters
- $r_i$: Revenue per unit of product $i \in I$ (from file_0_view_0, column Revenue).
- $d_i$: Demand for product $i \in I$ (from file_0_view_0, column Demand).
- $s_i$: Initial Inventory for product $i \in I$ (from file_0_view_0, column Initial Inventory).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective
$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. Inventory constraint:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand constraint:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- Set $I$ and all parameters ($r_i$, $d_i$, $s_i$) are defined from table_id file_0_view_0 in SalesStoreoverview.csv, using columns:
    - Product_Reference (for $I$)
    - Revenue (for $r_i$)
    - Demand (for $d_i$)
    - Initial Inventory (for $s_i$)
- The filter for $I$ is all rows where Product_Reference starts with "ELE-S".
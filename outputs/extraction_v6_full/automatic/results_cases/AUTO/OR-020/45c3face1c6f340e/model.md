#### Index Sets

- $I$: set of all products, indexed by $i$.

#### Parameters

- $r_i$: revenue per unit of product $i$.  
- $d_i$: total demand for product $i$.  
- $s_i$: initial inventory available for product $i$.

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints

1. Inventory constraints:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraints:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$ (product index set): All unique values in column "Product Name" from table_id file_0_view_0.
- $r_i$: "Revenue" column, indexed by "Product Name", from table_id file_0_view_0.
- $d_i$: "Demand" column, indexed by "Product Name", from table_id file_0_view_0.
- $s_i$: "Initial Inventory" column, indexed by "Product Name", from table_id file_0_view_0.

(CSVQA_DATA bindings: file_0_view_0, columns "Product Name", "Revenue", "Demand", "Initial Inventory")
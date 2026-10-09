#### Index Sets

- $I$: set of all clothing products, indexed by $i$ (from all rows in table_id = file_0_view_0, column "Product Name")

#### Parameters

- $A_i$: revenue per unit of product $i$ (from table_id = file_0_view_0, column "Revenue")
- $d_i$: deterministic demand for product $i$ (from table_id = file_0_view_0, column "Demand")
- $I_i$: initial inventory for product $i$ (from table_id = file_0_view_0, column "Initial Inventory")

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, integer, $x_i \geq 0$, for all $i \in I$

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraint for each product:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraint for each product:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$: All unique values in "Product Name" from table_id = file_0_view_0 (file: Salesofsummerclothes.csv)
- $A_i$: "Revenue" column, table_id = file_0_view_0
- $d_i$: "Demand" column, table_id = file_0_view_0
- $I_i$: "Initial Inventory" column, table_id = file_0_view_0
- $x_i$: Decision variable for each $i \in I$
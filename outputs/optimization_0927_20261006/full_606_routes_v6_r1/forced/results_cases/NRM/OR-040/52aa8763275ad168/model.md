#### Index Sets

- $I$: set of areas (from products.csv, column ProductName)

#### Parameters

- $b_i$: benefit coefficient for area $i \in I$ (from products.csv, column Value)
- $C$: overall development capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: integer, scale of development in area $i$ per day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. Capacity constraint:
   $$
   \sum_{i \in I} x_i \leq C
   $$

2. Integer and nonnegativity constraints:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv, column: Capacity, table_id: file_0_view_0 — provides parameter $C$ (overall development capacity).
- Table: products.csv, columns: ProductName, Value, table_id: file_1_view_0 — provides index set $I$ (areas) and parameter $b_i$ (benefit coefficient for each area $i$).
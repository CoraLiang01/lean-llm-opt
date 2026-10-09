#### Index Sets

- $I$: set of areas available for development (from products.csv, column ProductName)

#### Parameters

- $b_i$: development benefit per unit scale in area $i$ (from products.csv, column Value, for $i \in I$)
- $w_i$: resource usage per unit scale in area $i$ (from products.csv, column Weight, for $i \in I$)
- $C$: overall development capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: scale of development per day in area $i$, $x_i \geq 0$, continuous (for $i \in I$)

#### Objective

$$
\max \sum_{i \in I} b_i \cdot x_i
$$

#### Constraints

1. Overall Capacity Constraint:
   $$
   \sum_{i \in I} w_i \cdot x_i \leq C
   $$

2. Non-negativity:
   $$
   x_i \geq 0 \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv, column Capacity, table_id: file_0_view_0 — provides parameter $C$ (overall development capacity).
- Table: products.csv, columns ProductName, Value, Weight, table_id: file_1_view_0 — provides index set $I$ (areas), parameter $b_i$ (development benefit per unit), and parameter $w_i$ (resource usage per unit) for each area $i \in I$. Only areas with ProductName in {"Queens", "Brooklyn"} are included, as per the query and returned records.
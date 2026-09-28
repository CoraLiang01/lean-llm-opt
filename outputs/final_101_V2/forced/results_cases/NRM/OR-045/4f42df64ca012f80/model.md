#### Index Sets

- $I$: set of produce types (from products.csv, column ProductName)

#### Parameters

- $v_i$: benefit per unit of produce $i$ (from products.csv, column Value)
- $w_i$: weight per unit of produce $i$ (from products.csv, column Weight)
- $C$: total inventory capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: integer number of units of produce $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

#### Objective

$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraints

1. Capacity constraint:
   $$
   \sum_{i \in I} w_i x_i \leq C
   $$
2. Integer and non-negativity constraints:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv, column: Capacity, table_id: file_0_view_0 $\rightarrow$ parameter $C$
- Table: products.csv, columns: ProductName, Value, Weight, table_id: file_1_view_0 $\rightarrow$ index set $I$, parameters $v_i$, $w_i$
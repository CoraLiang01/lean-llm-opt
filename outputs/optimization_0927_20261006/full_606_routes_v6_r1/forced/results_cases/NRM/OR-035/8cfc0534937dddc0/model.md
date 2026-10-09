#### Index Sets

- $I$: set of bread types (from products.csv, column ProductName)

#### Parameters

- $v_i$: expected profit per unit of bread type $i$ (from products.csv, column Value)
- $w_i$: storage weight per unit of bread type $i$ (from products.csv, column Weight)
- $C$: total storage capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraints

1. Storage capacity:
   $$
   \sum_{i \in I} w_i x_i \leq C
   $$

2. Integer and non-negativity:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0), column: Capacity $\rightarrow C$
- Table: products.csv (table_id: file_1_view_0), columns: ProductName $\rightarrow I$, Value $\rightarrow v_i$, Weight $\rightarrow w_i$
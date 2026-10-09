#### Mathematical Model

Let:
- $I$ = set of produce types, indexed by $i$ (from all ProductName in products.csv)
- $x_i$ = number of units of produce $i$ to order daily (decision variable, integer, $x_i \geq 0$)
- $v_i$ = value (benefit) per unit of produce $i$
- $w_i$ = weight per unit of produce $i$
- $C$ = overall inventory capacity

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$: All records in products.csv, column ProductName (table_id: file_1_view_0, column: ProductName)
- $v_i$: products.csv, column Value (table_id: file_1_view_0, column: Value), mapped by ProductName
- $w_i$: products.csv, column Weight (table_id: file_1_view_0, column: Weight), mapped by ProductName
- $C$: capacity.csv, column Capacity (table_id: file_0_view_0, column: Capacity)
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)
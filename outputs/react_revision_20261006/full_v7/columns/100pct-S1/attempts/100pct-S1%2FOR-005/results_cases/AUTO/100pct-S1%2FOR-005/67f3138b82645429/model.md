##### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), as given in products.csv.
- $x_i$ = number of units of bread type $i$ to order each day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = expected profit per unit of bread type $i$ (from item_value).
- $a_i$ = storage space required per unit of bread type $i$ (from resource_requirement).
- $C$ = total storage capacity (from resource_capacity).

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: All item_name in table_id file_1_view_0 (products.csv)
- $v_i$: item_value in table_id file_1_view_0, mapped by item_name
- $a_i$: resource_requirement in table_id file_1_view_0, mapped by item_name
- $C$: resource_capacity in table_id file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (bread type)
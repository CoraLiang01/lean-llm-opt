#### Mathematical Model

Let $I$ be the set of bread types (indexed by $i$), with each $i$ corresponding to a unique item_name from products.csv.

Let $x_i$ = number of units of bread type $i$ to order each day.

Parameters:
- $v_i$ = expected profit per unit of bread type $i$ (from item_value)
- $a_i$ = storage space required per unit of bread type $i$ (from resource_requirement)
- $C$ = total available storage capacity (from resource_capacity)

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

---

#### Data Mapping

- $I$: All item_name in file_1_view_0 (products.csv)
- $v_i$: item_value column in file_1_view_0, mapped by item_name
- $a_i$: resource_requirement column in file_1_view_0, mapped by item_name
- $C$: resource_capacity in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each item_name in file_1_view_0

All parameters and index sets are defined directly from the returned tables, preserving original file and column names.
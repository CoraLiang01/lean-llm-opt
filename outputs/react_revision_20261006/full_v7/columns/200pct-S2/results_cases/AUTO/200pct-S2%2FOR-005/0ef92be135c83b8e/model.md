##### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), from all item_name in products.csv.
- $x_i$ = number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = expected profit per unit of bread type $i$ (item_value).
- $a_i$ = storage space required per unit of bread type $i$ (resource_requirement).
- $C$ = total storage capacity (resource_capacity).

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

##### Data Mapping

- $I$: All item_name in table_id file_1_view_0 (products.csv)
- $v_i$: item_value from table_id file_1_view_0, column item_value, for each $i$
- $a_i$: resource_requirement from table_id file_1_view_0, column resource_requirement, for each $i$
- $C$: resource_capacity from table_id file_0_view_0 (capacity.csv), column resource_capacity

All variables, parameters, and constraints are mapped directly to the columns and records as described above.
#### Abstract Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$, with business identifier item_name from products.csv.
- For each $i \in I$:
    - $v_i$ = expected profit per unit of bread $i$ (item_value from products.csv, table_id: file_1_view_0)
    - $a_i$ = storage requirement per unit of bread $i$ (resource_requirement from products.csv, table_id: file_1_view_0)
- $C$ = total available storage capacity (resource_capacity from capacity.csv, table_id: file_0_view_0)
- $x_i$ = number of units of bread $i$ to order each day (decision variable, integer, $\geq 0$)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} a_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (bread types): item_name from products.csv (table_id: file_1_view_0)
- $v_i$: item_value from products.csv (table_id: file_1_view_0)
- $a_i$: resource_requirement from products.csv (table_id: file_1_view_0)
- $C$: resource_capacity from capacity.csv (table_id: file_0_view_0)

All parameters and variables are mapped directly to the columns and business identifiers as returned by CSVQA.
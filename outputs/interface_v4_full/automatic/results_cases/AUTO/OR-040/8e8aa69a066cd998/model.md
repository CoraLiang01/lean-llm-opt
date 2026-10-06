#### Abstract Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area identifiers from products.csv ProductName.
- $x_i$ = integer variable: scale of development in area $i$ per day.

Parameters:
- $v_i$ = benefit coefficient for area $i$ (from products.csv Value).
- $w_i$ = development unit weight for area $i$ (from products.csv Weight).
- $C$ = overall development capacity (from capacity.csv Capacity).

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity, row 0.

All variables, parameters, and constraints are indexed and mapped exactly as above.
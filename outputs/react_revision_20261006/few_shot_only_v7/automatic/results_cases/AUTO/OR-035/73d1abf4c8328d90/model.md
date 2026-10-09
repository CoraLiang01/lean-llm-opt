##### Mathematical Model

Let $I$ be the set of bread types (indexed by $i$), as given by the ProductName column in products.csv.

Let $x_i$ = number of units of bread type $i$ to order each day.

Parameters:
- $p_i$ = expected profit per unit of bread $i$ (from Value column in products.csv)
- $w_i$ = storage space required per unit of bread $i$ (from Weight column in products.csv)
- $C$ = total storage capacity (from Capacity column in capacity.csv)

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

##### Data Mapping

- $I$: products.csv, ProductName
- $p_i$: products.csv, Value
- $w_i$: products.csv, Weight
- $C$: capacity.csv, Capacity
- $x_i$: decision variable for each $i \in I$ (bread type)
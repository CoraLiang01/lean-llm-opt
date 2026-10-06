ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of areas available for development (indexed by $i$), with area names as in products.csv.

Parameters:
- $b_i$: Development benefit per unit in area $i$ (from products.csv, column Value, key ProductName).
- $w_i$: Resource requirement per unit in area $i$ (from products.csv, column Weight, key ProductName).
- $C$: Overall development capacity (from capacity.csv, column Capacity).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$.

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping:

- $b_i$: products.csv, column Value, key ProductName
- $w_i$: products.csv, column Weight, key ProductName
- $C$: capacity.csv, column Capacity

All parameters and identifiers are used as returned, with no omitted or synthesized data.
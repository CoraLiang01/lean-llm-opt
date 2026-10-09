##### Mathematical Model

Let:
- $I$ = set of areas, indexed by $i$ (from products.csv, column ProductName)
- $b_i$ = benefit coefficient of area $i$ (products.csv, column Value)
- $w_i$ = development unit requirement for area $i$ (products.csv, column Weight)
- $C$ = total development capacity (capacity.csv, column Capacity)
- $x_i$ = integer number of development units in area $i$ per day

**Objective:**
$$
\max \sum_{i \in I} b_i x_i
$$

**Subject to:**
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

##### Data Mapping

- $I$: All records in products.csv, column ProductName
- $b_i$: products.csv, column Value, for each $i$
- $w_i$: products.csv, column Weight, for each $i$
- $C$: capacity.csv, column Capacity (single value)
- $x_i$: integer variable for each $i \in I$
##### Abstract Mathematical Model

Let:
- $I$ = set of drug products (indexed by $i$), from products.csv, column ProductName
- $x_i$ = number of units of drug $i$ to order each day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$
- $b_i$ = benefit per unit of drug $i$ (from products.csv, column Value)
- $w_i$ = weight (stock space requirement) per unit of drug $i$ (from products.csv, column Weight)
- $C$ = total stock capacity (from capacity.csv, column Capacity)

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

- $I$: products.csv, column ProductName
- $b_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C$: capacity.csv, column Capacity
- $x_i$: decision variable, number of units to order for each $i \in I$
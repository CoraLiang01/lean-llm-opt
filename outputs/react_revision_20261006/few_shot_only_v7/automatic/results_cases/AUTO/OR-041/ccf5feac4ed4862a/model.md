##### Mathematical Model

Let $I$ be the set of areas (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Value (development benefit) of area $i$ (from products.csv, column Value)
- $w_i$: Weight (resource requirement per unit development in area $i$) (from products.csv, column Weight)
- $C$: Total development capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of real estate units to develop in area $i$

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
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
- $v_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C$: capacity.csv, column Capacity (single value)
- $x_i$: Decision variable for each $i \in I$ (area)
**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of areas (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Development benefit of area $i$ (from products.csv, column Value)
- $w_i$: Resource requirement per unit scale in area $i$ (from products.csv, column Weight)
- $C$: Total development capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: products.csv, column ProductName
- $v_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C$: capacity.csv, column Capacity

---

**Notes:**
- All areas in products.csv are included in $I$.
- The model maximizes total development benefit subject to the overall development capacity.
- The scale of development per area is a nonnegative integer.
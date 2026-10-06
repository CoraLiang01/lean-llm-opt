**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of produce types (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Value (benefit) per unit of produce $i$ (from products.csv, column Value)
- $w_i$: Weight per unit of produce $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of produce $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

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

- $I$: All records in products.csv, column ProductName
- $v_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C$: capacity.csv, column Capacity (single value)
- $x_i$: Decision variable for each $i \in I$ (produce type)
**Abstract Mathematical Model**

Let:

- $I$ = set of vehicle types, indexed by $i$ (from products.csv, column ProductName)
- $x_i$ = number of units of vehicle type $i$ to order daily (integer, $x_i \geq 0$)
- $v_i$ = benefit coefficient of vehicle type $i$ (from products.csv, column Value)
- $w_i$ = inventory weight of vehicle type $i$ (from products.csv, column Weight)
- $C$ = total inventory capacity (from capacity.csv, column Capacity)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**

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
- $C$: capacity.csv, column Capacity (single record)
- $x_i$: Decision variable for each $i \in I$
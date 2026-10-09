##### Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from ProductName in products.csv)
- $x_i$ = number of units of bread type $i$ to order each day (integer, $x_i \geq 0$)
- $v_i$ = expected profit per unit of bread type $i$ (Value from products.csv)
- $w_i$ = weight (storage space required) per unit of bread type $i$ (Weight from products.csv)
- $C$ = total storage capacity (Capacity from capacity.csv)

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

##### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value

All variables, parameters, and constraints are mapped directly to the columns and records as returned above.
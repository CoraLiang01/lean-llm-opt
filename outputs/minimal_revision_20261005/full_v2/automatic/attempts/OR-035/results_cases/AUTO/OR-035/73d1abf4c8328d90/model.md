**Abstract Mathematical Model**

**Sets:**
- $I$: Set of bread types, indexed by $i$ (from products.csv)

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$ (from products.csv, column Value)
- $w_i$: Storage weight per unit of bread type $i$ (from products.csv, column Weight)
- $C$: Total storage capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each ProductName in file_1_view_0

---

**Notes:**
- All bread types in products.csv are included in $I$.
- The storage capacity $C$ is the single value from capacity.csv.
- All variables $x_i$ are nonnegative integers.
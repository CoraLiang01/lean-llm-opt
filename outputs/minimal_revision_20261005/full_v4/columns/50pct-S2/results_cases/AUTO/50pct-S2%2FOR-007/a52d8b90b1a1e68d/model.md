**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Profit per unit of vehicle $i$ (from products.csv, column Value)
- $w_i$: Inventory space required per unit of vehicle $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

---

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

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity, row 0

---

**Notes:**
- All vehicle types in products.csv are included in $I$.
- The total inventory capacity $C$ is the single value from capacity.csv.
- Each $x_i$ is a nonnegative integer, as required by the problem statement.
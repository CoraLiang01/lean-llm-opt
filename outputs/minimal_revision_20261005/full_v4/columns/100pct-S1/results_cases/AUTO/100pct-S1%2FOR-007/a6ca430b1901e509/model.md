**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Profit per unit of vehicle $i$ (from products.csv, column Value, for $i \in I$).
- $w_i$: Inventory space required per unit of vehicle $i$ (from products.csv, column Weight, for $i \in I$).
- $C$: Total inventory capacity (from capacity.csv, column Capacity).

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity.

**Notes:**
- All vehicle types in products.csv are included in $I$.
- The model maximizes total profit from daily vehicle orders, subject to the overall inventory capacity.
- All variables are nonnegative integers as required.
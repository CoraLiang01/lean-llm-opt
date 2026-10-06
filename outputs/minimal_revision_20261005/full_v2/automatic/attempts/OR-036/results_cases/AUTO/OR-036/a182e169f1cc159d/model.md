**Mathematical Optimization Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from `products.csv`, column `ProductName`).

**Parameters:**
- $v_i$: Benefit coefficient of vehicle type $i$ (from `products.csv`, column `Value`).
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from `products.csv`, column `Weight`).
- $C$: Total inventory capacity (from `capacity.csv`, column `Capacity`).

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily. ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All records in `file_1_view_0` (`products.csv`), column `ProductName`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity` (single value)

---

**Summary:**  
Choose integer order quantities $x_i$ for each vehicle type $i$ to maximize total benefit, subject to the total inventory capacity. All parameters and index sets are mapped directly from the provided CSV files as specified above.
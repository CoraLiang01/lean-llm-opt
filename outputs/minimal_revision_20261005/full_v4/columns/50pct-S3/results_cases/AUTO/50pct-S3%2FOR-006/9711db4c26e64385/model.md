**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from `products.csv`.

**Parameters**
- $v_i$: Value (benefit coefficient) of vehicle type $i$.  
  Data Mapping: `file_1_view_0`, column `Value`, keyed by `ProductName`.
- $w_i$: Weight (inventory space required) of vehicle type $i$.  
  Data Mapping: `file_1_view_0`, column `Weight`, keyed by `ProductName`.
- $C$: Total inventory capacity.  
  Data Mapping: `file_0_view_0`, column `Capacity`.

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ to order daily.

**Objective**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All `ProductName` in `file_1_view_0`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity` (single value)

---

**Summary:**  
Choose integer order quantities $x_i$ for each vehicle type $i$ to maximize total benefit, subject to the total inventory weight not exceeding $C$. All parameters and index sets are mapped directly from the provided CSV data.
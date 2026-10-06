**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from `products.csv`.

**Parameters**
- $v_i$: Value (benefit coefficient) of vehicle type $i$.  
  Data Mapping: `file_1_view_0`, column `Value`, keyed by `ProductName`.
- $w_i$: Weight (inventory space requirement) of vehicle type $i$.  
  Data Mapping: `file_1_view_0`, column `Weight`, keyed by `ProductName`.
- $C$: Total inventory capacity.  
  Data Mapping: `file_0_view_0`, column `Capacity`.

**Decision Variables**
- $x_i$: Number of units of vehicle type $i$ to order each day.  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), $\forall i \in I$.

**Objective**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All `ProductName` values from `file_1_view_0`.
- $v_i$: `Value` column in `file_1_view_0`, keyed by `ProductName`.
- $w_i$: `Weight` column in `file_1_view_0`, keyed by `ProductName`.
- $C$: `Capacity` value from `file_0_view_0`, row 0.

---

**Summary:**  
Choose integer order quantities $x_i$ for each vehicle type $i \in I$ to maximize total benefit, subject to the total inventory capacity $C$. All parameters and index sets are mapped directly from the provided CSV data.
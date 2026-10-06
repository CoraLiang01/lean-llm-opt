**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of vehicle types (from file_1_view_0, column ProductName)

**Parameters**
- $p_i$: Profit per unit of vehicle $i$ (file_1_view_0, column Value)
- $w_i$: Inventory weight per unit of vehicle $i$ (file_1_view_0, column Weight)
- $C$: Total inventory capacity (file_0_view_0, column Capacity)

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day

**Objective**
\[
\max \sum_{i \in I} p_i x_i
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

- $I$: All records in file_1_view_0, column ProductName
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity, row 0

---

**Summary:**  
Choose integer order quantities $x_i$ for each vehicle type $i$ to maximize total profit, subject to the overall inventory capacity. All parameters and index sets are mapped directly from the supplied CSV data.
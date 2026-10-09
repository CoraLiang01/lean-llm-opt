### Abstract Mathematical Model

**Index Sets:**
- $I$: set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $p_i$: profit per unit of vehicle $i$ (from products.csv, column Value)
- $w_i$: inventory weight per unit of vehicle $i$ (from products.csv, column Weight)
- $C$: total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)
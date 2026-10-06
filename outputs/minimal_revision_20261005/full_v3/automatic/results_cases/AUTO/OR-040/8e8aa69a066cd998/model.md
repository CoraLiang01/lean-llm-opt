**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of areas (from products.csv, column ProductName)

**Parameters**
- $v_i$: Benefit coefficient for area $i$ (from products.csv, column Value)
- $w_i$: Development unit weight for area $i$ (from products.csv, column Weight)
- $C$: Overall development capacity (from capacity.csv, column Capacity)

**Decision Variables**
- $x_i$: Integer number of development units in area $i$ per day

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

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity, row 0

**Variables**
- $x_i$: Integer, $\geq 0$, for each $i \in I$ (ProductName from file_1_view_0)
**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of drug types, indexed by $i$. (from file_1_view_0, column ProductName)

**Parameters**
- $v_i$: Benefit coefficient of drug type $i$. (file_1_view_0, column Value)
- $w_i$: Weight per unit of drug type $i$. (file_1_view_0, column Weight)
- $C$: Total inventory capacity. (file_0_view_0, column Capacity)

**Decision Variables**
- $x_i$: Number of units of drug type $i$ to order daily. ($x_i \in \mathbb{Z}_{\geq 0}$)

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
- $x_i$: Decision variable for each $i \in I$
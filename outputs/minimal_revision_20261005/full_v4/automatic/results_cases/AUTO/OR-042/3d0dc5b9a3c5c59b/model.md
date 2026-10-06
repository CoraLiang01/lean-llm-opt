**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of drug types, indexed by $i$. (from products.csv, column ProductName)

**Parameters**
- $v_i$: Benefit coefficient of drug type $i$. (from products.csv, column Value)
- $w_i$: Weight per unit of drug type $i$. (from products.csv, column Weight)
- $C$: Total inventory capacity. (from capacity.csv, column Capacity)

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity.

**Notes**
- All parameters and index sets are defined exactly as returned by CSVQA.
- Each $x_i$ is a nonnegative integer variable representing the daily order quantity for drug type $i$.
- The model maximizes total benefit subject to the overall inventory weight capacity.
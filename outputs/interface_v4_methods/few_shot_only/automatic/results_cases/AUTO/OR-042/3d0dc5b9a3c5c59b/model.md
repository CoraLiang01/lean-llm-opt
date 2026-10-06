#### Abstract Mathematical Model

**Index Sets:**
- Let $I$ be the set of drug types, indexed by $i$, with identifiers ProductName from file_1_view_0.

**Parameters:**
- $v_i$: benefit coefficient of drug $i$ (Value column in file_1_view_0)
- $w_i$: weight per unit of drug $i$ (Weight column in file_1_view_0)
- $C$: total inventory capacity (Capacity column in file_0_view_0)

**Decision Variables:**
- $x_i$: number of units of drug $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

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

#### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), ProductName
- $v_i$: file_1_view_0, Value, for each ProductName
- $w_i$: file_1_view_0, Weight, for each ProductName
- $C$: file_0_view_0, Capacity

**Source Tables Used:**
- file_0_view_0: capacity.csv, columns [Capacity]
- file_1_view_0: products.csv, columns [ProductName, Value, Weight]
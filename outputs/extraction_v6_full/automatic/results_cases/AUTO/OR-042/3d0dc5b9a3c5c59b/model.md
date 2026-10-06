#### Abstract Mathematical Model

Let $I$ be the set of drug types (indexed by $i$), with each $i \in I$ corresponding to a unique ProductName from products.csv.

**Parameters:**
- $b_i$: benefit coefficient of drug $i$ (from Value column)
- $w_i$: weight per unit of drug $i$ (from Weight column)
- $C$: overall inventory capacity (from Capacity column)

**Decision Variables:**
- $x_i$: number of units of drug $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv), in source order.
- $b_i$: file_1_view_0, column Value, indexed by ProductName.
- $w_i$: file_1_view_0, column Weight, indexed by ProductName.
- $C$: file_0_view_0, column Capacity.

Each parameter and index set is mapped directly to its source table and column as above. No data is omitted or synthesized.
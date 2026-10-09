#### Mathematical Model

Let $I$ be the set of produce types, indexed by $i$ (from the "ProductName" column in products.csv).

**Decision variables:**
- $x_i$: number of units of produce $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$

**Parameters:**
- $v_i$: value (benefit) per unit of produce $i$ (from "Value" in products.csv)
- $w_i$: weight per unit of produce $i$ (from "Weight" in products.csv)
- $C$: total inventory capacity (from "Capacity" in capacity.csv)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
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

- $I$: All records in file_1_view_0 (products.csv), column "ProductName"
- $v_i$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_i$: file_1_view_0, column "Weight", keyed by "ProductName"
- $C$: file_0_view_0, column "Capacity"
- $x_i$: decision variable for each $i \in I$ (produce type)
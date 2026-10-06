**Mathematical Optimization Model**

**Index Sets:**
- $I$: Set of products, indexed by $i$. (From `file_1_view_0`, column `ProductName`)

**Parameters:**
- $v_i$: Value (benefit) per unit of product $i$. (From `file_1_view_0`, column `Value`)
- $w_i$: Weight (stock space required) per unit of product $i$. (From `file_1_view_0`, column `Weight`)
- $C$: Total stock capacity. (From `file_0_view_0`, column `Capacity`)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

---

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

**Data Mapping:**

- $I$: All records in `file_1_view_0`, column `ProductName`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity`, row 0

---

**Notes:**
- All products and their parameters are taken directly from the supplied rows in `products.csv`.
- The total stock capacity $C$ is taken from the single row in `capacity.csv`.
- The model maximizes total benefit from daily orders, subject to the overall stock capacity.
- All decision variables are nonnegative integers, as required.
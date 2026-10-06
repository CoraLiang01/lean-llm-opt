**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of areas (from `file_1_view_0`, column `ProductName`)
- (There is only one overall capacity constraint, not per-area.)

**Parameters:**
- $v_i$: Development benefit of area $i$ (from `file_1_view_0`, column `Value`)
- $w_i$: Resource requirement per unit development in area $i$ (from `file_1_view_0`, column `Weight`)
- $C$: Overall development capacity (from `file_0_view_0`, column `Capacity`)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

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

**Data Mapping**

- $I$: All records in `file_1_view_0`, column `ProductName`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity` (single value)

---

**Notes:**
- All areas in `file_1_view_0` are included in $I$.
- The model maximizes total development benefit subject to the overall development capacity.
- Each $x_i$ is a nonnegative integer, as required.
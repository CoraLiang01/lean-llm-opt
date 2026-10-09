##### Mathematical Model

Let:
- $I$ = set of produce types, indexed by $i$ (from all ProductName in products.csv)
- $x_i$ = number of units of produce $i$ to order daily (integer, $\geq 0$)
- $v_i$ = value (benefit) per unit of produce $i$
- $w_i$ = weight per unit of produce $i$
- $C$ = overall inventory capacity

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

##### Data Mapping

- $I$: All records in `file_1_view_0` (products.csv), column `ProductName`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity`
- $x_i$: Decision variable for each $i \in I$ (produce type)
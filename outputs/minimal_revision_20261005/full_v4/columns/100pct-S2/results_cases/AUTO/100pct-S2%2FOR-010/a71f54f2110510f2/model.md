**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of sections, indexed by $s$ (from all SectionID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: Capacity of section $s$ (file_0_view_0, column: Capacity, key: SectionID)
- $v_p$: Value (revenue) per unit of product $p$ (file_1_view_0, column: Value, key: ProductName)
- $w_p$: Space requirement per unit of product $p$ (file_1_view_0, column: Weight, key: ProductName)

**Decision Variables:**
- $x_{s,p}$: Number of units of product $p$ to stock in section $s$; $x_{s,p} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Subject to:**

1. **Section Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{s,p} \leq c_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All SectionID from file_0_view_0 (capacity.csv), column: SectionID
- $P$: All ProductName from file_1_view_0 (products.csv), column: ProductName
- $c_s$: file_0_view_0, column: Capacity, key: SectionID
- $v_p$: file_1_view_0, column: Value, key: ProductName
- $w_p$: file_1_view_0, column: Weight, key: ProductName

**Variable Domain:**
- $x_{s,p}$: integer, $\geq 0$ (nonnegative integer)
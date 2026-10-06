**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of sections, indexed by $s$ (from all SectionID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of section $s$ (from Capacity in file_0_view_0, indexed by SectionID)
- $v_p$: Value (revenue) per unit of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: Space requirement per unit of product $p$ (from Weight in file_1_view_0, indexed by ProductName)

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
   \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All SectionID in `file_0_view_0`, column `SectionID`
- $P$: All ProductName in `file_1_view_0`, column `ProductName`
- $C_s$: `file_0_view_0`, columns `SectionID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`
- $x_{s,p}$: Decision variable for each $(s,p) \in S \times P$
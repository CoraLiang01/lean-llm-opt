**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of sections, indexed by $s$ (from all SectionID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: display space capacity of section $s$ (from file_0_view_0, column Capacity, keyed by SectionID)
- $v_p$: value (revenue) per unit of product $p$ (from file_1_view_0, column Value, keyed by ProductName)
- $w_p$: shelf space requirement per unit of product $p$ (from file_1_view_0, column Weight, keyed by ProductName)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Section Capacity Constraints:**  
   For each section $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq c_s
   \]

2. **Integrality and Nonnegativity:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All SectionID from `file_0_view_0`, column SectionID
- $P$: All ProductName from `file_1_view_0`, column ProductName
- $c_s$: `file_0_view_0`, columns SectionID (key), Capacity (value)
- $v_p$: `file_1_view_0`, columns ProductName (key), Value (value)
- $w_p$: `file_1_view_0`, columns ProductName (key), Weight (value)
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$
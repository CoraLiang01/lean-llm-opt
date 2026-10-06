**Sets:**
- $S$: Set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of section $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value (revenue) per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: Space requirement per unit of product $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ to stock in section $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

- **Section Capacity Constraints:**
  \[
  \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
  \]

- **Integrality and Nonnegativity:**
  \[
  x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
  \]

---

### Data Mapping

- $S$ (sections): file_0_view_0, column SectionID
- $C_s$: file_0_view_0, columns SectionID, Capacity
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, columns ProductName, Value
- $w_p$: file_1_view_0, columns ProductName, Weight

---

**Summary:**  
Maximize total revenue from product allocations to sections, subject to each section's display space limit. Decision variables are nonnegative integers representing the number of units of each product in each section. All parameters and sets are mapped directly to the supplied CSV columns and business identifiers.
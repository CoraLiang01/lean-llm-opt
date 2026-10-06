**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: Set of products, indexed by $p$ (ProductName from file_1_view_0)

**Parameters:**
- $C_s$: Capacity (display space limit) of section $s$ (file_0_view_0, column "Capacity")
- $v_p$: Value (price) of product $p$ (file_1_view_0, column "Value")
- $w_p$: Shelf space requirement (weight) of product $p$ (file_1_view_0, column "Weight")

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

1. **Section Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All SectionID values from file_0_view_0 (capacity.csv), column "SectionID"
- $P$: All ProductName values from file_1_view_0 (products.csv), column "ProductName"
- $C_s$: file_0_view_0 (capacity.csv), columns: SectionID, Capacity
- $v_p$: file_1_view_0 (products.csv), columns: ProductName, Value
- $w_p$: file_1_view_0 (products.csv), columns: ProductName, Weight

---

**Summary:**  
Choose integer quantities $x_{sp}$ of each product $p$ for each section $s$ to maximize total revenue, subject to each section's display space limit. All parameters and index sets are mapped directly from the supplied data tables.
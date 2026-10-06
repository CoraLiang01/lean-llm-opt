---

### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of section $s$ (from file_0_view_0, column Capacity, key SectionID)
- $v_p$: Value (price) of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: Space requirement of product $p$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ to stock in section $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Section Capacity Constraints:**  
   For each section $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

### Data Mapping

**Section Capacity ($C_s$):**  
From `file_0_view_0` (capacity.csv), use column `SectionID` as $s$ and column `Capacity` as $C_s$.

**Product Value ($v_p$) and Space Requirement ($w_p$):**  
From `file_1_view_0` (products.csv), use column `ProductName` as $p$, column `Value` as $v_p$, and column `Weight` as $w_p$.

**Decision Variables:**  
$x_{sp}$ is defined for every combination of $s$ in SectionID (file_0_view_0) and $p$ in ProductName (file_1_view_0).

---

**All parameters and index sets are to be populated directly from the referenced columns and business IDs in the retrieved CSV data.**
ABSTRACT MATHEMATICAL MODEL

**Index Sets:**
- $S$: set of section IDs (from `file_0_view_0.SectionID`)
- $P$: set of product names (from `file_1_view_0.ProductName`)

**Parameters:**
- $C_s$: capacity of section $s$ (from `file_0_view_0.Capacity`)
- $v_p$: value (price) of product $p$ (from `file_1_view_0.Value`)
- $w_p$: shelf space requirement of product $p$ (from `file_1_view_0.Weight`)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

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
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping:**

- $S$: All `SectionID` in `file_0_view_0`
- $P$: All `ProductName` in `file_1_view_0`
- $C_s$: `file_0_view_0.Capacity` for section $s$
- $v_p$: `file_1_view_0.Value` for product $p$
- $w_p$: `file_1_view_0.Weight` for product $p$
- $x_{sp}$: Number of units of product $p$ in section $s$ (decision variable)
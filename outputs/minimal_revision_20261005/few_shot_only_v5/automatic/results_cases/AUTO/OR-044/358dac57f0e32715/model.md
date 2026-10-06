**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of store sections (indexed by $s$), from capacity.csv [SectionID]
- $P$: Set of products (indexed by $p$), from products.csv [ProductName]

**Parameters:**
- $c_s$: Capacity (display space limit) of section $s$; from capacity.csv [Capacity]
- $v_p$: Price (revenue per unit) of product $p$; from products.csv [Value]
- $w_p$: Shelf space requirement per unit of product $p$; from products.csv [Weight]

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
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]
2. **Integrality and Nonnegativity:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: capacity.csv [SectionID]
- $c_s$: capacity.csv [Capacity], keyed by [SectionID]
- $P$: products.csv [ProductName]
- $v_p$: products.csv [Value], keyed by [ProductName]
- $w_p$: products.csv [Weight], keyed by [ProductName]
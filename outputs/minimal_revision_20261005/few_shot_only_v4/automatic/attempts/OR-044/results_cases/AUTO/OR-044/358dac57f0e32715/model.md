**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of store sections (indexed by $s$), from capacity.csv, column SectionID
- $P$: Set of products (indexed by $p$), from products.csv, column ProductName

**Parameters:**
- $c_s$: Capacity (display space limit) of section $s$ (capacity.csv, column Capacity)
- $v_p$: Value (price) of product $p$ (products.csv, column Value)
- $w_p$: Shelf space requirement of product $p$ (products.csv, column Weight)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: capacity.csv, column SectionID
- $P$: products.csv, column ProductName
- $c_s$: capacity.csv, column Capacity, keyed by SectionID
- $v_p$: products.csv, column Value, keyed by ProductName
- $w_p$: products.csv, column Weight, keyed by ProductName
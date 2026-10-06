**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $v_p$: Value per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight per unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ allocated to shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
   \]
2. **Integrality and Nonnegativity:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

---

**Summary:**  
Maximize total product value allocated to shelves, subject to each shelf's capacity, using integer allocation variables for each product-shelf pair. All parameters and index sets are mapped directly from the supplied CSV data.
**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)
- $v_p$: value of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: weight of product $p$ (file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{s,p} \leq c_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID from file_0_view_0 (capacity.csv)
- $P$: All ProductName from file_1_view_0 (products.csv)
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

**Notes:**  
- All products and shelves from the returned data are included.
- Each $x_{s,p}$ is a nonnegative integer, as required.
- Each shelf’s total product weight cannot exceed its capacity.  
- All parameters are mapped directly to the original data columns and keys.
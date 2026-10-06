**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of storage areas (indexed by $s$), from capacity.csv [StorageID]
- $P$: Set of air conditioner types (indexed by $p$), from products.csv [ProductName]

**Parameters:**
- $C_s$: Capacity of storage area $s$, from capacity.csv [Capacity]
- $v_p$: Value of air conditioner type $p$, from products.csv [Value]
- $w_p$: Weight (size) of air conditioner type $p$, from products.csv [Weight]

**Decision Variables:**
- $x_{sp}$: Number of units of air conditioner type $p$ placed in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**

1. **Storage Area Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s, \quad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: capacity.csv [StorageID]
- $C_s$: capacity.csv [Capacity], keyed by [StorageID]
- $P$: products.csv [ProductName]
- $v_p$: products.csv [Value], keyed by [ProductName]
- $w_p$: products.csv [Weight], keyed by [ProductName]
- $x_{sp}$: Decision variable for allocation of product $p$ to storage area $s$
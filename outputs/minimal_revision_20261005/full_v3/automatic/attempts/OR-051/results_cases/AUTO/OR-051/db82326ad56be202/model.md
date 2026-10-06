**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of cabinets, indexed by $i$ (from file_0_view_0, column CabinetID)
- $J$: Set of coffee products, indexed by $j$ (from file_1_view_0, column ProductName)

**Parameters:**
- $c_i$: Capacity of cabinet $i$ (from file_0_view_0, column Capacity)
- $v_j$: Value per unit of product $j$ (from file_1_view_0, column Value)
- $w_j$: Weight per unit of product $j$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Cabinet Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$: All values in file_0_view_0, column CabinetID
- $J$: All values in file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by CabinetID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

**End of Model**
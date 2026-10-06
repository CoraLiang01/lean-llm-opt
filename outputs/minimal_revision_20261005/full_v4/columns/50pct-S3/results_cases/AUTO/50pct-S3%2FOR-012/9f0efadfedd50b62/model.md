**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of platforms, indexed by $p$ (from all resource_id in file_0_view_0)
- $G$: Set of game genres, indexed by $g$ (from all item_name in file_1_view_0)

**Parameters:**
- $C_p$: Memory capacity of platform $p$ (resource_capacity from file_0_view_0, keyed by resource_id)
- $v_g$: Value per unit of genre $g$ (item_value from file_1_view_0, keyed by item_name)
- $r_g$: Memory requirement per unit of genre $g$ (resource_requirement from file_1_view_0, keyed by item_name)

**Decision Variables:**
- $x_{pg}$: Number of units of games from genre $g$ to be listed on platform $p$; $x_{pg} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{p \in P} \sum_{g \in G} v_g \cdot x_{pg}
\]

**Constraints:**
1. **Platform Memory Capacity:**
   \[
   \sum_{g \in G} r_g \cdot x_{pg} \leq C_p \qquad \forall p \in P
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{pg} \in \mathbb{Z}_{\geq 0} \qquad \forall p \in P,\, g \in G
   \]

---

**Data Mapping**

- $P$: All resource_id in file_0_view_0 (capacity.csv)
- $G$: All item_name in file_1_view_0 (products.csv)
- $C_p$: file_0_view_0.resource_capacity, keyed by file_0_view_0.resource_id
- $v_g$: file_1_view_0.item_value, keyed by file_1_view_0.item_name
- $r_g$: file_1_view_0.resource_requirement, keyed by file_1_view_0.item_name
- $x_{pg}$: Decision variable for platform $p$ (file_0_view_0.resource_id) and genre $g$ (file_1_view_0.item_name)
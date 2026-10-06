**Abstract Mathematical Model**

**Index Sets:**
- $P$: set of platforms, indexed by $p$ (from all resource_id in file_0_view_0)
- $G$: set of game genres, indexed by $g$ (from all item_name in file_1_view_0)

**Parameters:**
- $C_p$: memory capacity of platform $p$ (resource_capacity from file_0_view_0, keyed by resource_id)
- $v_g$: value per unit of genre $g$ (item_value from file_1_view_0, keyed by item_name)
- $r_g$: memory requirement per unit of genre $g$ (resource_requirement from file_1_view_0, keyed by item_name)

**Decision Variables:**
- $x_{pg}$: number of units of games from genre $g$ to be listed on platform $p$; $x_{pg} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{p \in P} \sum_{g \in G} v_g \cdot x_{pg}
\]

**Constraints:**
1. **Platform Memory Capacity:**
   \[
   \sum_{g \in G} r_g \cdot x_{pg} \leq C_p \qquad \forall p \in P
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{pg} \in \mathbb{Z}_{\geq 0} \qquad \forall p \in P,\, g \in G
   \]

---

**Data Mapping**

- $P$: All resource_id in file_0_view_0 (capacity.csv)
- $G$: All item_name in file_1_view_0 (products.csv)
- $C_p$: resource_capacity in file_0_view_0, keyed by resource_id
- $v_g$: item_value in file_1_view_0, keyed by item_name
- $r_g$: resource_requirement in file_1_view_0, keyed by item_name

**Source Tables Used:**
- file_0_view_0: columns [resource_id, resource_capacity] (capacity.csv)
- file_1_view_0: columns [item_name, item_value, resource_requirement] (products.csv)
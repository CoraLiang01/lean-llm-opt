**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of platforms, indexed by $p$ (from all resource_id in file_0_view_0)
- $G$: Set of game genres, indexed by $g$ (from all item_name in file_1_view_0)

**Parameters:**
- $C_p$: Memory capacity of platform $p$ (resource_capacity from file_0_view_0, indexed by resource_id)
- $v_g$: Value per unit of genre $g$ (item_value from file_1_view_0, indexed by item_name)
- $r_g$: Memory requirement per unit of genre $g$ (resource_requirement from file_1_view_0, indexed by item_name)

**Decision Variables:**
- $x_{p,g}$: Number of units of games from genre $g$ to list on platform $p$; $x_{p,g} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{p \in P} \sum_{g \in G} v_g \cdot x_{p,g}
\]

**Subject to:**

- **Platform Memory Capacity Constraints:**
  \[
  \sum_{g \in G} r_g \cdot x_{p,g} \leq C_p, \quad \forall p \in P
  \]

- **Integrality and Nonnegativity:**
  \[
  x_{p,g} \in \mathbb{Z}_{\geq 0}, \quad \forall p \in P,\, g \in G
  \]

---

**Data Mapping**

- $P$: All resource_id in file_0_view_0 (capacity.csv)
- $G$: All item_name in file_1_view_0 (products.csv)
- $C_p$: resource_capacity from file_0_view_0, indexed by resource_id
- $v_g$: item_value from file_1_view_0, indexed by item_name
- $r_g$: resource_requirement from file_1_view_0, indexed by item_name

**Variable Mapping**

- $x_{p,g}$: Number of units of games from genre $g$ to list on platform $p$ (integer, $\geq 0$)
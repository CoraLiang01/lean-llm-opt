#### Abstract Mathematical Model

**Index Sets:**
- $P$: Set of platforms (indexed by $p$), from `capacity.csv` column `PlatformID`.
- $G$: Set of games (indexed by $g$), from `products.csv` column `ProductName`.

**Parameters:**
- $C_p$: Memory capacity of platform $p$, from `capacity.csv` column `Capacity`.
- $v_g$: Value of game $g$, from `products.csv` column `Value`.
- $w_g$: Memory requirement of game $g$, from `products.csv` column `Weight$.

**Decision Variables:**
- $x_{pg} \in \mathbb{Z}_{\geq 0}$: Number of units of game $g$ to be listed on platform $p$.

**Objective:**
\[
\max \sum_{p \in P} \sum_{g \in G} v_g \cdot x_{pg}
\]

**Constraints:**
1. **Platform Memory Capacity:**
   \[
   \sum_{g \in G} w_g \cdot x_{pg} \leq C_p \qquad \forall p \in P
   \]
2. **Integrality:**
   \[
   x_{pg} \in \mathbb{Z}_{\geq 0} \qquad \forall p \in P,\, g \in G
   \]

---

#### Data Mapping

- $P$: `capacity.csv` (`file_0_view_0`), column `PlatformID`
- $C_p$: `capacity.csv` (`file_0_view_0`), column `Capacity`, keyed by `PlatformID`
- $G$: `products.csv` (`file_1_view_0`), column `ProductName`
- $v_g$: `products.csv` (`file_1_view_0`), column `Value`, keyed by `ProductName`
- $w_g$: `products.csv` (`file_1_view_0`), column `Weight`, keyed by `ProductName`
- $x_{pg}$: Number of units of game $g$ on platform $p$ (decision variable, integer, nonnegative)
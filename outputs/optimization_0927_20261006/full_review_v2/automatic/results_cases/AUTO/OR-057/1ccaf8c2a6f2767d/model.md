Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let $P$ be the set of platforms, indexed by PlatformID as in the data (1 to 15).
Let $G$ be the set of game genres, indexed by ProductName as in the data.

Let $v_j$ be the Value of genre $j$ (from products.csv).
Let $w_j$ be the Weight (memory requirement) of genre $j$ (from products.csv).
Let $C_i$ be the Capacity of platform $i$ (from capacity.csv).

**Objective:**
\[
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i \in P$:
\[
\sum_{j \in G} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in P$, $j \in G$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Parameter values (in source order):**

Platforms and capacities:
- PlatformID 1: $C_1 = 995$
- PlatformID 2: $C_2 = 1143$
- PlatformID 3: $C_3 = 949$
- PlatformID 4: $C_4 = 969$
- PlatformID 5: $C_5 = 1649$
- PlatformID 6: $C_6 = 870$
- PlatformID 7: $C_7 = 1064$
- PlatformID 8: $C_8 = 536$
- PlatformID 9: $C_9 = 766$
- PlatformID 10: $C_{10} = 532$
- PlatformID 11: $C_{11} = 1703$
- PlatformID 12: $C_{12} = 1633$
- PlatformID 13: $C_{13} = 1203$
- PlatformID 14: $C_{14} = 1979$
- PlatformID 15: $C_{15} = 1797$

Game genres, values, and weights:
- Racing: $v_{\text{Racing}} = 59$, $w_{\text{Racing}} = 776$
- Sports: $v_{\text{Sports}} = 83$, $w_{\text{Sports}} = 573$
- Action: $v_{\text{Action}} = 94$, $w_{\text{Action}} = 127$
- Adventure: $v_{\text{Adventure}} = 41$, $w_{\text{Adventure}} = 138$
- RPG: $v_{\text{RPG}} = 96$, $w_{\text{RPG}} = 385$
- Shooter: $v_{\text{Shooter}} = 12$, $w_{\text{Shooter}} = 263$
- Strategy: $v_{\text{Strategy}} = 83$, $w_{\text{Strategy}} = 473$
- Simulation: $v_{\text{Simulation}} = 36$, $w_{\text{Simulation}} = 387$
- Puzzle: $v_{\text{Puzzle}} = 56$, $w_{\text{Puzzle}} = 390$
- Fighting: $v_{\text{Fighting}} = 27$, $w_{\text{Fighting}} = 556$
- Platformer: $v_{\text{Platformer}} = 47$, $w_{\text{Platformer}} = 601$
- Survival: $v_{\text{Survival}} = 24$, $w_{\text{Survival}} = 441$
- Horror: $v_{\text{Horror}} = 14$, $w_{\text{Horror}} = 603$
- Sandbox: $v_{\text{Sandbox}} = 22$, $w_{\text{Sandbox}} = 411$
- MMO: $v_{\text{MMO}} = 17$, $w_{\text{MMO}} = 652$

**Decision variables:**
- $x_{ij}$: integer, $\geq 0$, for all $i \in \{1,\ldots,15\}$, $j \in$ {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{15} \sum_{j \in G} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in G} w_j \cdot x_{ij} \leq C_i, \quad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,15, \; j \in G
\end{align*}
\]

where $G$ and all parameter values are as listed above.
Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:

- $i$ index platforms, with resource_id from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ index genres, with item_name from products.csv: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$

Let $v_j$ be the value (item_value) of genre $j$.

Let $a_j$ be the memory requirement (resource_requirement) of genre $j$.

Let $c_i$ be the memory capacity (resource_capacity) of platform $i$.

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
$$

where

\[
\begin{align*}
v_{\text{Racing}} &= 28 \\
v_{\text{Sports}} &= 69 \\
v_{\text{Action}} &= 20 \\
v_{\text{Adventure}} &= 62 \\
v_{\text{RPG}} &= 58 \\
v_{\text{Shooter}} &= 11 \\
v_{\text{Strategy}} &= 73 \\
v_{\text{Simulation}} &= 43 \\
v_{\text{Puzzle}} &= 28 \\
v_{\text{Fighting}} &= 57 \\
v_{\text{Platformer}} &= 92 \\
v_{\text{Survival}} &= 66 \\
v_{\text{Horror}} &= 14 \\
v_{\text{Sandbox}} &= 49 \\
v_{\text{MMO}} &= 12 \\
\end{align*}
\]

---

**Constraints:**

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

where

\[
\begin{align*}
a_{\text{Racing}} &= 393 \\
a_{\text{Sports}} &= 195 \\
a_{\text{Action}} &= 192 \\
a_{\text{Adventure}} &= 155 \\
a_{\text{RPG}} &= 500 \\
a_{\text{Shooter}} &= 156 \\
a_{\text{Strategy}} &= 317 \\
a_{\text{Simulation}} &= 694 \\
a_{\text{Puzzle}} &= 751 \\
a_{\text{Fighting}} &= 467 \\
a_{\text{Platformer}} &= 796 \\
a_{\text{Survival}} &= 146 \\
a_{\text{Horror}} &= 269 \\
a_{\text{Sandbox}} &= 246 \\
a_{\text{MMO}} &= 652 \\
\end{align*}
\]

and

\[
\begin{align*}
c_1 &= 1336 \\
c_2 &= 1754 \\
c_3 &= 1617 \\
c_4 &= 1119 \\
c_5 &= 1410 \\
c_6 &= 627 \\
c_7 &= 748 \\
c_8 &= 1540 \\
c_9 &= 1292 \\
c_{10} &= 1138 \\
\end{align*}
\]

---

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{15} a_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,15
\end{align*}
\]

with $v_j$, $a_j$, $c_i$ as specified above, and genre-to-index mapping as per the order in products.csv.
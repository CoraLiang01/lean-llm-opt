Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $i$ index platforms (resource_id from capacity.csv), and $j$ index genres (item_name from products.csv).

Let $v_j$ be the item_value of genre $j$.

Let $a_j$ be the resource_requirement (memory requirement) of genre $j$.

Let $c_i$ be the resource_capacity of platform $i$.

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
$$

where

- $v_{\text{Racing}} = 28$
- $v_{\text{Sports}} = 69$
- $v_{\text{Action}} = 20$
- $v_{\text{Adventure}} = 62$
- $v_{\text{RPG}} = 58$
- $v_{\text{Shooter}} = 11$
- $v_{\text{Strategy}} = 73$
- $v_{\text{Simulation}} = 43$
- $v_{\text{Puzzle}} = 28$
- $v_{\text{Fighting}} = 57$
- $v_{\text{Platformer}} = 92$
- $v_{\text{Survival}} = 66$
- $v_{\text{Horror}} = 14$
- $v_{\text{Sandbox}} = 49$
- $v_{\text{MMO}} = 12$

---

**Subject to:**

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i
$$

where

- $a_{\text{Racing}} = 393$
- $a_{\text{Sports}} = 195$
- $a_{\text{Action}} = 192$
- $a_{\text{Adventure}} = 155$
- $a_{\text{RPG}} = 500$
- $a_{\text{Shooter}} = 156$
- $a_{\text{Strategy}} = 317$
- $a_{\text{Simulation}} = 694$
- $a_{\text{Puzzle}} = 751$
- $a_{\text{Fighting}} = 467$
- $a_{\text{Platformer}} = 796$
- $a_{\text{Survival}} = 146$
- $a_{\text{Horror}} = 269$
- $a_{\text{Sandbox}} = 246$
- $a_{\text{MMO}} = 652$

and

- $c_1 = 1336$
- $c_2 = 1754$
- $c_3 = 1617$
- $c_4 = 1119$
- $c_5 = 1410$
- $c_6 = 627$
- $c_7 = 748$
- $c_8 = 1540$
- $c_9 = 1292$
- $c_{10} = 1138$

For all $i \in \{1,\ldots,10\}$ and $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Complete Model:**

Maximize
$$
\sum_{i=1}^{10} \Big[ 28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \Big]
$$

Subject to, for each $i$:

$$
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i
$$

with $c_i$ as above for each platform.

and

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
$$
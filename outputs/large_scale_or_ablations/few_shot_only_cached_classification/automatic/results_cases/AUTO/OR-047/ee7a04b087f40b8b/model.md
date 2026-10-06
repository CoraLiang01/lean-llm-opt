Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Parameters:**

- Platforms $i$ (from capacity.csv, in order):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
  with capacities $C_i$:
  - $C_1 = 1336$
  - $C_2 = 1754$
  - $C_3 = 1617$
  - $C_4 = 1119$
  - $C_5 = 1410$
  - $C_6 = 627$
  - $C_7 = 748$
  - $C_8 = 1540$
  - $C_9 = 1292$
  - $C_{10} = 1138$

- Genres $j$ (from products.csv, in order), with value $v_j$ and memory requirement $w_j$:
    1. Racing: $v_1 = 28$, $w_1 = 393$
    2. Sports: $v_2 = 69$, $w_2 = 195$
    3. Action: $v_3 = 20$, $w_3 = 192$
    4. Adventure: $v_4 = 62$, $w_4 = 155$
    5. RPG: $v_5 = 58$, $w_5 = 500$
    6. Shooter: $v_6 = 11$, $w_6 = 156$
    7. Strategy: $v_7 = 73$, $w_7 = 317$
    8. Simulation: $v_8 = 43$, $w_8 = 694$
    9. Puzzle: $v_9 = 28$, $w_9 = 751$
    10. Fighting: $v_{10} = 57$, $w_{10} = 467$
    11. Platformer: $v_{11} = 92$, $w_{11} = 796$
    12. Survival: $v_{12} = 66$, $w_{12} = 146$
    13. Horror: $v_{13} = 14$, $w_{13} = 269$
    14. Sandbox: $v_{14} = 49$, $w_{14} = 246$
    15. MMO: $v_{15} = 12$, $w_{15} = 652$

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \, x_{ij}
\]

**Subject to:**

For each platform $i = 1, \ldots, 10$:
\[
\sum_{j=1}^{15} w_j \, x_{ij} \leq C_i
\]

For all $i = 1, \ldots, 10$, $j = 1, \ldots, 15$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (integer, $\geq 0$)
- $v_j$ = value of genre $j$ (see above)
- $w_j$ = memory requirement of genre $j$ (see above)
- $C_i$ = memory capacity of platform $i$ (see above)

All identifiers and coefficients are as retrieved and in original order.
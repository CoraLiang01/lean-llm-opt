Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Parameters:**

- Platforms (from capacity.csv, in order):

  1. resource_id = 1, resource_capacity = 1336
  2. resource_id = 2, resource_capacity = 1754
  3. resource_id = 3, resource_capacity = 1617
  4. resource_id = 4, resource_capacity = 1119
  5. resource_id = 5, resource_capacity = 1410
  6. resource_id = 6, resource_capacity = 627
  7. resource_id = 7, resource_capacity = 748
  8. resource_id = 8, resource_capacity = 1540
  9. resource_id = 9, resource_capacity = 1292
  10. resource_id = 10, resource_capacity = 1138

- Genres (from products.csv, in order):

  1. Racing: item_value = 28, resource_requirement = 393
  2. Sports: item_value = 69, resource_requirement = 195
  3. Action: item_value = 20, resource_requirement = 192
  4. Adventure: item_value = 62, resource_requirement = 155
  5. RPG: item_value = 58, resource_requirement = 500
  6. Shooter: item_value = 11, resource_requirement = 156
  7. Strategy: item_value = 73, resource_requirement = 317
  8. Simulation: item_value = 43, resource_requirement = 694
  9. Puzzle: item_value = 28, resource_requirement = 751
  10. Fighting: item_value = 57, resource_requirement = 467
  11. Platformer: item_value = 92, resource_requirement = 796
  12. Survival: item_value = 66, resource_requirement = 146
  13. Horror: item_value = 14, resource_requirement = 269
  14. Sandbox: item_value = 49, resource_requirement = 246
  15. MMO: item_value = 12, resource_requirement = 652

---

**Mathematical Model**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,15\}
$$

where $x_{ij}$ is the number of units of games from genre $j$ to be listed on platform $i$.

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value for genre $j$ (see table above).

**Constraints:**

For each platform $i$ (resource_id as above):

$$
\sum_{j=1}^{15} r_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where:
- $r_j$ is the resource_requirement (memory requirement) for genre $j$,
- $C_i$ is the resource_capacity for platform $i$.

**Explicitly, for each platform:**

- For resource_id = 1:
  $$
  393x_{1,1} + 195x_{1,2} + 192x_{1,3} + 155x_{1,4} + 500x_{1,5} + 156x_{1,6} + 317x_{1,7} + 694x_{1,8} + 751x_{1,9} + 467x_{1,10} + 796x_{1,11} + 146x_{1,12} + 269x_{1,13} + 246x_{1,14} + 652x_{1,15} \leq 1336
  $$
- For resource_id = 2:
  $$
  393x_{2,1} + 195x_{2,2} + 192x_{2,3} + 155x_{2,4} + 500x_{2,5} + 156x_{2,6} + 317x_{2,7} + 694x_{2,8} + 751x_{2,9} + 467x_{2,10} + 796x_{2,11} + 146x_{2,12} + 269x_{2,13} + 246x_{2,14} + 652x_{2,15} \leq 1754
  $$
- For resource_id = 3:
  $$
  393x_{3,1} + 195x_{3,2} + 192x_{3,3} + 155x_{3,4} + 500x_{3,5} + 156x_{3,6} + 317x_{3,7} + 694x_{3,8} + 751x_{3,9} + 467x_{3,10} + 796x_{3,11} + 146x_{3,12} + 269x_{3,13} + 246x_{3,14} + 652x_{3,15} \leq 1617
  $$
- For resource_id = 4:
  $$
  393x_{4,1} + 195x_{4,2} + 192x_{4,3} + 155x_{4,4} + 500x_{4,5} + 156x_{4,6} + 317x_{4,7} + 694x_{4,8} + 751x_{4,9} + 467x_{4,10} + 796x_{4,11} + 146x_{4,12} + 269x_{4,13} + 246x_{4,14} + 652x_{4,15} \leq 1119
  $$
- For resource_id = 5:
  $$
  393x_{5,1} + 195x_{5,2} + 192x_{5,3} + 155x_{5,4} + 500x_{5,5} + 156x_{5,6} + 317x_{5,7} + 694x_{5,8} + 751x_{5,9} + 467x_{5,10} + 796x_{5,11} + 146x_{5,12} + 269x_{5,13} + 246x_{5,14} + 652x_{5,15} \leq 1410
  $$
- For resource_id = 6:
  $$
  393x_{6,1} + 195x_{6,2} + 192x_{6,3} + 155x_{6,4} + 500x_{6,5} + 156x_{6,6} + 317x_{6,7} + 694x_{6,8} + 751x_{6,9} + 467x_{6,10} + 796x_{6,11} + 146x_{6,12} + 269x_{6,13} + 246x_{6,14} + 652x_{6,15} \leq 627
  $$
- For resource_id = 7:
  $$
  393x_{7,1} + 195x_{7,2} + 192x_{7,3} + 155x_{7,4} + 500x_{7,5} + 156x_{7,6} + 317x_{7,7} + 694x_{7,8} + 751x_{7,9} + 467x_{7,10} + 796x_{7,11} + 146x_{7,12} + 269x_{7,13} + 246x_{7,14} + 652x_{7,15} \leq 748
  $$
- For resource_id = 8:
  $$
  393x_{8,1} + 195x_{8,2} + 192x_{8,3} + 155x_{8,4} + 500x_{8,5} + 156x_{8,6} + 317x_{8,7} + 694x_{8,8} + 751x_{8,9} + 467x_{8,10} + 796x_{8,11} + 146x_{8,12} + 269x_{8,13} + 246x_{8,14} + 652x_{8,15} \leq 1540
  $$
- For resource_id = 9:
  $$
  393x_{9,1} + 195x_{9,2} + 192x_{9,3} + 155x_{9,4} + 500x_{9,5} + 156x_{9,6} + 317x_{9,7} + 694x_{9,8} + 751x_{9,9} + 467x_{9,10} + 796x_{9,11} + 146x_{9,12} + 269x_{9,13} + 246x_{9,14} + 652x_{9,15} \leq 1292
  $$
- For resource_id = 10:
  $$
  393x_{10,1} + 195x_{10,2} + 192x_{10,3} + 155x_{10,4} + 500x_{10,5} + 156x_{10,6} + 317x_{10,7} + 694x_{10,8} + 751x_{10,9} + 467x_{10,10} + 796x_{10,11} + 146x_{10,12} + 269x_{10,13} + 246x_{10,14} + 652x_{10,15} \leq 1138
  $$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,15\}
$$

**Summary of genre parameters (in source order):**

| $j$ | Genre        | $v_j$ (item_value) | $r_j$ (resource_requirement) |
|-----|--------------|--------------------|-----------------------------|
| 1   | Racing       | 28                 | 393                         |
| 2   | Sports       | 69                 | 195                         |
| 3   | Action       | 20                 | 192                         |
| 4   | Adventure    | 62                 | 155                         |
| 5   | RPG          | 58                 | 500                         |
| 6   | Shooter      | 11                 | 156                         |
| 7   | Strategy     | 73                 | 317                         |
| 8   | Simulation   | 43                 | 694                         |
| 9   | Puzzle       | 28                 | 751                         |
| 10  | Fighting     | 57                 | 467                         |
| 11  | Platformer   | 92                 | 796                         |
| 12  | Survival     | 66                 | 146                         |
| 13  | Horror       | 14                 | 269                         |
| 14  | Sandbox      | 49                 | 246                         |
| 15  | MMO          | 12                 | 652                         |

**Summary of platform parameters (in source order):**

| $i$ | resource_id | $C_i$ (resource_capacity) |
|-----|-------------|--------------------------|
| 1   | 1           | 1336                     |
| 2   | 2           | 1754                     |
| 3   | 3           | 1617                     |
| 4   | 4           | 1119                     |
| 5   | 5           | 1410                     |
| 6   | 6           | 627                      |
| 7   | 7           | 748                      |
| 8   | 8           | 1540                     |
| 9   | 9           | 1292                     |
| 10  | 10          | 1138                     |

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{15} r_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,15
\end{align*}
$$

with all $v_j$, $r_j$, and $C_i$ as specified above, and all indices and coefficients in the original source order.
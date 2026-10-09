Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $i$ index platforms, with platform IDs as follows (from capacity.csv, in order):

- $i=1$: resource_id = 1, resource_capacity = 1336
- $i=2$: resource_id = 2, resource_capacity = 1754
- $i=3$: resource_id = 3, resource_capacity = 1617
- $i=4$: resource_id = 4, resource_capacity = 1119
- $i=5$: resource_id = 5, resource_capacity = 1410
- $i=6$: resource_id = 6, resource_capacity = 627
- $i=7$: resource_id = 7, resource_capacity = 748
- $i=8$: resource_id = 8, resource_capacity = 1540
- $i=9$: resource_id = 9, resource_capacity = 1292
- $i=10$: resource_id = 10, resource_capacity = 1138

Let $j$ index genres, with genre names as follows (from products.csv, in order):

1. Racing: value = 28, memory requirement = 393
2. Sports: value = 69, memory requirement = 195
3. Action: value = 20, memory requirement = 192
4. Adventure: value = 62, memory requirement = 155
5. RPG: value = 58, memory requirement = 500
6. Shooter: value = 11, memory requirement = 156
7. Strategy: value = 73, memory requirement = 317
8. Simulation: value = 43, memory requirement = 694
9. Puzzle: value = 28, memory requirement = 751
10. Fighting: value = 57, memory requirement = 467
11. Platformer: value = 92, memory requirement = 796
12. Survival: value = 66, memory requirement = 146
13. Horror: value = 14, memory requirement = 269
14. Sandbox: value = 49, memory requirement = 246
15. MMO: value = 12, memory requirement = 652

Define:

- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$
- $c_i$ = resource_capacity of platform $i$

The model is:

$$
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
$$

Subject to, for each platform $i=1,\ldots,10$:

$$
\sum_{j=1}^{15} a_j x_{ij} \leq c_i
$$

and

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\; j=1,\ldots,15
$$

Where:

- For $i=1$: $c_1 = 1336$
- For $i=2$: $c_2 = 1754$
- For $i=3$: $c_3 = 1617$
- For $i=4$: $c_4 = 1119$
- For $i=5$: $c_5 = 1410$
- For $i=6$: $c_6 = 627$
- For $i=7$: $c_7 = 748$
- For $i=8$: $c_8 = 1540$
- For $i=9$: $c_9 = 1292$
- For $i=10$: $c_{10} = 1138$

And for $j=1$ to $15$:

| $j$ | Genre        | $v_j$ | $a_j$ |
|-----|-------------|-------|-------|
| 1   | Racing      | 28    | 393   |
| 2   | Sports      | 69    | 195   |
| 3   | Action      | 20    | 192   |
| 4   | Adventure   | 62    | 155   |
| 5   | RPG         | 58    | 500   |
| 6   | Shooter     | 11    | 156   |
| 7   | Strategy    | 73    | 317   |
| 8   | Simulation  | 43    | 694   |
| 9   | Puzzle      | 28    | 751   |
| 10  | Fighting    | 57    | 467   |
| 11  | Platformer  | 92    | 796   |
| 12  | Survival    | 66    | 146   |
| 13  | Horror      | 14    | 269   |
| 14  | Sandbox     | 49    | 246   |
| 15  | MMO         | 12    | 652   |

All variables $x_{ij}$ are integer and $\geq 0$.
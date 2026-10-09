Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (platforms, identified by resource_id)
- $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (genres, identified by item_name)

Parameters:
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$
- $c_i$ = resource_capacity of platform $i$

Numerical values from the data:

Platforms (resource_id, resource_capacity):
- 1: 1336
- 2: 1754
- 3: 1617
- 4: 1119
- 5: 1410
- 6: 627
- 7: 748
- 8: 1540
- 9: 1292
- 10: 1138

Genres (item_name, item_value, resource_requirement):
- Racing: 28, 393
- Sports: 69, 195
- Action: 20, 192
- Adventure: 62, 155
- RPG: 58, 500
- Shooter: 11, 156
- Strategy: 73, 317
- Simulation: 43, 694
- Puzzle: 28, 751
- Fighting: 57, 467
- Platformer: 92, 796
- Survival: 66, 146
- Horror: 14, 269
- Sandbox: 49, 246
- MMO: 12, 652

The complete mathematical model is:

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]
where $v_j$ is as above.

Subject to, for each platform $i$:
\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $a_j$ and $c_i$ are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Explicitly, with all coefficients:

\[
\max \Bigg[
\sum_{i=1}^{10} \Big(
28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}}
\Big)
\Bigg]
\]

For each $i$ (platform/resource_id):

\[
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i
\]
where $c_i$ is:
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

And for all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]
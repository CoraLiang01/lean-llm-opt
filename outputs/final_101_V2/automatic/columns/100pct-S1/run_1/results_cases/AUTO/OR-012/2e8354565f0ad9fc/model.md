Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Indices:
- $i$ indexes platforms, with resource_id from "capacity.csv".
- $j$ indexes game genres, with item_name from "products.csv".

Parameters:
- $c_i$: resource_capacity of platform $i$ (from "capacity.csv").
- $v_j$: item_value of genre $j$ (from "products.csv").
- $a_j$: resource_requirement of genre $j$ (from "products.csv").

Sets:
- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (resource_id)
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (item_name)

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
$$

Subject to:

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

Variable domains:

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Numerical Data:

Platforms (from "capacity.csv"):
- resource_id: 1, resource_capacity: 1336
- resource_id: 2, resource_capacity: 1754
- resource_id: 3, resource_capacity: 1617
- resource_id: 4, resource_capacity: 1119
- resource_id: 5, resource_capacity: 1410
- resource_id: 6, resource_capacity: 627
- resource_id: 7, resource_capacity: 748
- resource_id: 8, resource_capacity: 1540
- resource_id: 9, resource_capacity: 1292
- resource_id: 10, resource_capacity: 1138

Genres (from "products.csv"):
- Racing: item_value = 28, resource_requirement = 393
- Sports: item_value = 69, resource_requirement = 195
- Action: item_value = 20, resource_requirement = 192
- Adventure: item_value = 62, resource_requirement = 155
- RPG: item_value = 58, resource_requirement = 500
- Shooter: item_value = 11, resource_requirement = 156
- Strategy: item_value = 73, resource_requirement = 317
- Simulation: item_value = 43, resource_requirement = 694
- Puzzle: item_value = 28, resource_requirement = 751
- Fighting: item_value = 57, resource_requirement = 467
- Platformer: item_value = 92, resource_requirement = 796
- Survival: item_value = 66, resource_requirement = 146
- Horror: item_value = 14, resource_requirement = 269
- Sandbox: item_value = 49, resource_requirement = 246
- MMO: item_value = 12, resource_requirement = 652

Complete Model:

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
$$

Subject to, for each $i$:

$$
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i
$$

where $c_i$ is the resource_capacity for platform $i$ as listed above.

And

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$
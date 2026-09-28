##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game $j \in J$ (genre) to be listed on platform $i \in I$.

##### Parameters

- $I = \{1,2,\ldots,15\}$ (platforms)
- $J = \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (games/genres)
- Platform capacities:
  - $C_1 = 995$
  - $C_2 = 1143$
  - $C_3 = 949$
  - $C_4 = 969$
  - $C_5 = 1649$
  - $C_6 = 870$
  - $C_7 = 1064$
  - $C_8 = 536$
  - $C_9 = 766$
  - $C_{10} = 532$
  - $C_{11} = 1703$
  - $C_{12} = 1633$
  - $C_{13} = 1203$
  - $C_{14} = 1979$
  - $C_{15} = 1797$
- Game values and memory requirements:
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

##### Objective Function

$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

1. Platform memory capacity:
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   $$
2. Integer and nonnegativity:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   $$

##### Retrieved Information

{
  "platforms": {
    "1": 995,
    "2": 1143,
    "3": 949,
    "4": 969,
    "5": 1649,
    "6": 870,
    "7": 1064,
    "8": 536,
    "9": 766,
    "10": 532,
    "11": 1703,
    "12": 1633,
    "13": 1203,
    "14": 1979,
    "15": 1797
  },
  "games": {
    "Racing": {"Value": 59, "Weight": 776},
    "Sports": {"Value": 83, "Weight": 573},
    "Action": {"Value": 94, "Weight": 127},
    "Adventure": {"Value": 41, "Weight": 138},
    "RPG": {"Value": 96, "Weight": 385},
    "Shooter": {"Value": 12, "Weight": 263},
    "Strategy": {"Value": 83, "Weight": 473},
    "Simulation": {"Value": 36, "Weight": 387},
    "Puzzle": {"Value": 56, "Weight": 390},
    "Fighting": {"Value": 27, "Weight": 556},
    "Platformer": {"Value": 47, "Weight": 601},
    "Survival": {"Value": 24, "Weight": 441},
    "Horror": {"Value": 14, "Weight": 603},
    "Sandbox": {"Value": 22, "Weight": 411},
    "MMO": {"Value": 17, "Weight": 652}
  }
}
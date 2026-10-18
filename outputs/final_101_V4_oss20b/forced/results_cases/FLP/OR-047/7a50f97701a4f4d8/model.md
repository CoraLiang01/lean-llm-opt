##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of game genre $j$ listed on platform $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Platform memory capacity: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\leq C_i,\quad \forall i\in I$.

2. Integer non‑negativity: $x_{ij}\in \mathbb{Z}_{\ge 0},\quad \forall i\in I,\;j\in J$.

##### Sets and Parameters

$I=\{1,2,3,4,5,6,7,8,9,10\}$ (platform identifiers)  
$J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ (genre identifiers)

$C=\bigl[1336,\;1754,\;1617,\;1119,\;1410,\;627,\;748,\;1540,\;1292,\;1138\bigr]$ (memory capacities of platforms 1–10)

$v=\bigl[28,\;69,\;20,\;62,\;58,\;11,\;73,\;43,\;28,\;57,\;92,\;66,\;14,\;49,\;12\bigr]$ (values per unit of each genre)

$w=\bigl[393,\;195,\;192,\;155,\;500,\;156,\;317,\;694,\;751,\;467,\;796,\;146,\;269,\;246,\;652\bigr]$ (memory weight per unit of each genre)

##### Retrieved Information

```json
{
  "capacity": {
    "1": 1336,
    "2": 1754,
    "3": 1617,
    "4": 1119,
    "5": 1410,
    "6": 627,
    "7": 748,
    "8": 1540,
    "9": 1292,
    "10": 1138
  },
  "products": [
    {"ProductName":"Racing","Value":28,"Weight":393},
    {"ProductName":"Sports","Value":69,"Weight":195},
    {"ProductName":"Action","Value":20,"Weight":192},
    {"ProductName":"Adventure","Value":62,"Weight":155},
    {"ProductName":"RPG","Value":58,"Weight":500},
    {"ProductName":"Shooter","Value":11,"Weight":156},
    {"ProductName":"Strategy","Value":73,"Weight":317},
    {"ProductName":"Simulation","Value":43,"Weight":694},
    {"ProductName":"Puzzle","Value":28,"Weight":751},
    {"ProductName":"Fighting","Value":57,"Weight":467},
    {"ProductName":"Platformer","Value":92,"Weight":796},
    {"ProductName":"Survival","Value":66,"Weight":146},
    {"ProductName":"Horror","Value":14,"Weight":269},
    {"ProductName":"Sandbox","Value":49,"Weight":246},
    {"ProductName":"MMO","Value":12,"Weight":652}
  ]
}
```
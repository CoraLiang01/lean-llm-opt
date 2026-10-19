##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of game genre $j$ listed on platform $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Memory capacity on each platform:  
   $\displaystyle \sum_{j\in J}w_{j}\,x_{ij}\le c_{i},\quad\forall i\in I$  

2. Non‑negativity and integrality:  
   $\displaystyle x_{ij}\in \mathbb{Z}_{\ge 0},\quad\forall i\in I,\;j\in J$  

$ I=\{1,2,3,4,5,6,7,8,9,10\}$ (platform identifiers)  
$ J=\{\text{Racing},\text{Sports},\text{Action},\text{Adventure},\text{RPG},\text{Shooter},\text{Strategy},\text{Simulation},\text{Puzzle},\text{Fighting},\text{Platformer},\text{Survival},\text{Horror},\text{Sandbox},\text{MMO}\}$ (game genres)  

###### Retrieved Information

```json
{
  "platforms": [
    {"PlatformId":"1","Capacity":1336},
    {"PlatformId":"2","Capacity":1754},
    {"PlatformId":"3","Capacity":1617},
    {"PlatformId":"4","Capacity":1119},
    {"PlatformId":"5","Capacity":1410},
    {"PlatformId":"6","Capacity":627},
    {"PlatformId":"7","Capacity":748},
    {"PlatformId":"8","Capacity":1540},
    {"PlatformId":"9","Capacity":1292},
    {"PlatformId":"10","Capacity":1138}
  ],
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
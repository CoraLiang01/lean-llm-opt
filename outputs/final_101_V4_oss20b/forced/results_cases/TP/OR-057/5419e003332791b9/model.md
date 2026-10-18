##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\geq0}$: number of units of game $j$ listed on platform $i$.

##### Objective Function

$\max\sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Memory capacity: $\sum_{j\in J}w_{j}\,x_{ij}\leq c_{i},\quad\forall i\in I$  
2. Non‑negativity and integrality: $x_{ij}\in \mathbb{Z}_{\geq0},\quad\forall i\in I,\;j\in J$  

$I=\{1,2,\dots ,15\}$ (platform identifiers)  
$J=\{\text{Racing},\text{Sports},\text{Action},\text{Adventure},\text{RPG},\text{Shooter},\text{Strategy},\text{Simulation},\text{Puzzle},\text{Fighting},\text{Platformer},\text{Survival},\text{Horror},\text{Sandbox},\text{MMO}\}$  

###### Retrieved Information

{
  "platforms": [
    {"PlatformID":"1","Capacity":"995"},
    {"PlatformID":"2","Capacity":"1143"},
    {"PlatformID":"3","Capacity":"949"},
    {"PlatformID":"4","Capacity":"969"},
    {"PlatformID":"5","Capacity":"1649"},
    {"PlatformID":"6","Capacity":"870"},
    {"PlatformID":"7","Capacity":"1064"},
    {"PlatformID":"8","Capacity":"536"},
    {"PlatformID":"9","Capacity":"766"},
    {"PlatformID":"10","Capacity":"532"},
    {"PlatformID":"11","Capacity":"1703"},
    {"PlatformID":"12","Capacity":"1633"},
    {"PlatformID":"13","Capacity":"1203"},
    {"PlatformID":"14","Capacity":"1979"},
    {"PlatformID":"15","Capacity":"1797"}
  ],
  "products": [
    {"ProductName":"Racing","Value":"59","Weight":"776"},
    {"ProductName":"Sports","Value":"83","Weight":"573"},
    {"ProductName":"Action","Value":"94","Weight":"127"},
    {"ProductName":"Adventure","Value":"41","Weight":"138"},
    {"ProductName":"RPG","Value":"96","Weight":"385"},
    {"ProductName":"Shooter","Value":"12","Weight":"263"},
    {"ProductName":"Strategy","Value":"83","Weight":"473"},
    {"ProductName":"Simulation","Value":"36","Weight":"387"},
    {"ProductName":"Puzzle","Value":"56","Weight":"390"},
    {"ProductName":"Fighting","Value":"27","Weight":"556"},
    {"ProductName":"Platformer","Value":"47","Weight":"601"},
    {"ProductName":"Survival","Value":"24","Weight":"441"},
    {"ProductName":"Horror","Value":"14","Weight":"603"},
    {"ProductName":"Sandbox","Value":"22","Weight":"411"},
    {"ProductName":"MMO","Value":"17","Weight":"652"}
  ]
}
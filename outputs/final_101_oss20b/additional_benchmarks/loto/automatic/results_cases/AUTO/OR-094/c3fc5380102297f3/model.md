##### Objective Function:

$\quad \quad \min \sum_{j=1}^{3} \left( C_j - \sum_{i=1}^{101} t_{ij}\,x_i \right)$  

where  

$C_1 = 1296,\; C_2 = 1238.4,\; C_3 = 1267.2$  

and $t_{ij}$ is the processing time (minutes) of model $i$ on workstation $j$.

##### Constraints

###### 1. Capacity Constraints  

$\quad \quad \sum_{i=1}^{101} t_{ij}\,x_i \;\le\; C_j,\qquad \forall j=1,2,3$

###### 2. Non‑negativity and Integrality Constraints  

$\quad \quad x_i \;\in\; \mathbb{Z}_{\ge 0},\qquad \forall i=1,\dots,101$

##### Retrieved Information

**Effective Capacities**

```
{
  "Workstation1": 1296,
  "Workstation2": 1238.4,
  "Workstation3": 1267.2
}
```

**Processing Times (minutes) for each HiFi model on each workstation**

```
[
  {"Model":"HiFi1","Workstation1":6,"Workstation2":5,"Workstation3":4},
  {"Model":"HiFi2","Workstation1":4,"Workstation2":5,"Workstation3":6},
  {"Model":"HiFi3","Workstation1":6,"Workstation2":5,"Workstation3":5},
  {"Model":"HiFi4","Workstation1":7,"Workstation2":1,"Workstation3":2},
  {"Model":"HiFi5","Workstation1":6,"Workstation2":7,"Workstation3":6},
  {"Model":"HiFi6","Workstation1":6,"Workstation2":8,"Workstation3":5},
  {"Model":"HiFi7","Workstation1":8,"Workstation2":7,"Workstation3":3},
  {"Model":"HiFi8","Workstation1":9,"Workstation2":5,"Workstation3":3},
  {"Model":"HiFi9","Workstation1":6,"Workstation2":6,"Workstation3":4},
  {"Model":"HiFi10","Workstation1":7,"Workstation2":8,"Workstation3":8},
  {"Model":"HiFi11","Workstation1":1,"Workstation2":9,"Workstation3":6},
  {"Model":"HiFi12","Workstation1":2,"Workstation2":9,"Workstation3":3},
  {"Model":"HiFi13","Workstation1":4,"Workstation2":2,"Workstation3":3},
  {"Model":"HiFi14","Workstation1":7,"Workstation2":6,"Workstation3":3},
  {"Model":"HiFi15","Workstation1":3,"Workstation2":9,"Workstation3":7},
  {"Model":"HiFi16","Workstation1":8,"Workstation2":4,"Workstation3":8},
  {"Model":"HiFi17","Workstation1":3,"Workstation2":1,"Workstation3":3},
  {"Model":"HiFi18","Workstation1":2,"Workstation2":2,"Workstation3":8},
  {"Model":"HiFi19","Workstation1":4,"Workstation2":9,"Workstation3":1},
  {"Model":"HiFi20","Workstation1":5,"Workstation2":3,"Workstation3":5},
  {"Model":"HiFi21","Workstation1":8,"Workstation2":8,"Workstation3":3},
  {"Model":"HiFi22","Workstation1":3,"Workstation2":5,"Workstation3":8},
  {"Model":"HiFi23","Workstation1":2,"Workstation2":9,"Workstation3":5},
  {"Model":"HiFi24","Workstation1":3,"Workstation2":5,"Workstation3":8},
  {"Model":"HiFi25","Workstation1":9,"Workstation2":8,"Workstation3":4},
  {"Model":"HiFi26","Workstation1":7,"Workstation2":7,"Workstation3":8},
  {"Model":"HiFi27","Workstation1":3,"Workstation2":1,"Workstation3":6},
  {"Model":"HiFi28","Workstation1":5,"Workstation2":1,"Workstation3":7},
  {"Model":"HiFi29","Workstation1":7,"Workstation2":9,"Workstation3":9},
  {"Model":"HiFi30","Workstation1":6,"Workstation2":7,"Workstation3":5},
  {"Model":"HiFi31","Workstation1":2,"Workstation2":1,"Workstation3":3},
  {"Model":"HiFi32","Workstation1":1,"Workstation2":9,"Workstation3":6},
  {"Model":"HiFi33","Workstation1":5,"Workstation2":6,"Workstation3":3},
  {"Model":"HiFi34","Workstation1":6,"Workstation2":4,"Workstation3":3},
  {"Model":"HiFi35","Workstation1":5,"Workstation2":7,"Workstation3":3},
  {"Model":"HiFi36","Workstation1":1,"Workstation2":4,"Workstation3":8},
  {"Model":"HiFi37","Workstation1":7,"Workstation2":8,"Workstation3":4},
  {"Model":"HiFi38","Workstation1":9,"Workstation2":6,"Workstation3":6},
  {"Model":"HiFi39","Workstation1":8,"Workstation2":5,"Workstation3":3},
  {"Model":"HiFi40","Workstation1":3,"Workstation2":3,"Workstation3":8},
  {"Model":"HiFi41","Workstation1":3,"Workstation2":6,"Workstation3":3},
  {"Model":"HiFi42","Workstation1":8,"Workstation2":7,"Workstation3":7},
  {"Model":"HiFi43","Workstation1":2,"Workstation2":6,"Workstation3":5},
  {"Model":"HiFi44","Workstation1":3,"Workstation2":2,"Workstation3":3},
  {"Model":"HiFi45","Workstation1":3,"Workstation2":1,"Workstation3":1},
  {"Model":"HiFi46","Workstation1":8,"Workstation2":1,"Workstation3":8},
  {"Model":"HiFi47","Workstation1":9,"Workstation2":3,"Workstation3":9},
  {"Model":"HiFi48","Workstation1":2,"Workstation2":8,"Workstation3":6},
  {"Model":"HiFi49","Workstation1":3,"Workstation2":4,"Workstation3":6},
  {"Model":"HiFi50","Workstation1":4,"Workstation2":3,"Workstation3":4},
  {"Model":"HiFi51","Workstation1":2,"Workstation2":6,"Workstation3":7},
  {"Model":"HiFi52","Workstation1":9,"Workstation2":9,"Workstation3":1},
  {"Model":"HiFi53","Workstation1":2,"Workstation2":8,"Workstation3":9},
  {"Model":"HiFi54","Workstation1":1,"Workstation2":7,"Workstation3":9},
  {"Model":"HiFi55","Workstation1":8,"Workstation2":2,"Workstation3":3},
  {"Model":"HiFi56","Workstation1":8,"Workstation2":2,"Workstation3":9},
  {"Model":"HiFi57","Workstation1":4,"Workstation2":5,"Workstation3":6},
  {"Model":"HiFi58","Workstation1":4,"Workstation2":4,"Workstation3":5},
  {"Model":"HiFi59","Workstation1":6,"Workstation2":3,"Workstation3":7},
  {"Model":"HiFi60","Workstation1":1,"Workstation2":8,"Workstation3":8},
  {"Model":"HiFi61","Workstation1":6,"Workstation2":8,"Workstation3":9},
  {"Model":"HiFi62","Workstation1":5,"Workstation2":6,"Workstation3":9},
  {"Model":"HiFi63","Workstation1":3,"Workstation2":6,"Workstation3":8},
  {"Model":"HiFi64","Workstation1":5,"Workstation2":3,"Workstation3":5},
  {"Model":"HiFi65","Workstation1":1,"Workstation2":1,"Workstation3":4},
  {"Model":"HiFi66","Workstation1":6,"Workstation2":6,"Workstation3":4},
  {"Model":"HiFi67","Workstation1":6,"Workstation2":2,"Workstation3":3},
  {"Model":"HiFi68","Workstation1":5,"Workstation2":6,"Workstation3":3},
  {"Model":"HiFi69","Workstation1":3,"Workstation2":1,"Workstation3":8},
  {"Model":"HiFi70","Workstation1":4,"Workstation2":3,"Workstation3":8},
  {"Model":"HiFi71","Workstation1":3,"Workstation2":7,"Workstation3":2},
  {"Model":"HiFi72","Workstation1":8,"Workstation2":1,"Workstation3":4},
  {"Model":"HiFi73","Workstation1":1,"Workstation2":1,"Workstation3":9},
  {"Model":"HiFi74","Workstation1":2,"Workstation2":2,"Workstation3":6},
  {"Model":"HiFi75","Workstation1":3,"Workstation2":8,"Workstation3":7},
  {"Model":"HiFi76","Workstation1":2,"Workstation2":7,"Workstation3":6},
  {"Model":"HiFi77","Workstation1":8,"Workstation2":8,"Workstation3":7},
  {"Model":"HiFi78","Workstation1":4,"Workstation2":8,"Workstation3":3},
  {"Model":"HiFi79","Workstation1":4,"Workstation2":7,"Workstation3":1},
  {"Model":"HiFi80","Workstation1":2,"Workstation2":5,"Workstation3":7},
  {"Model":"HiFi81","Workstation1":7,"Workstation2":2,"Workstation3":6},
  {"Model":"HiFi82","Workstation1":5,"Workstation2":5,"Workstation3":4},
  {"Model":"HiFi83","Workstation1":1,"Workstation2":6,"Workstation3":3},
  {"Model":"HiFi84","Workstation1":6,"Workstation2":2,"Workstation3":5},
  {"Model":"HiFi85","Workstation1":4,"Workstation2":3,"Workstation3":7},
  {"Model":"HiFi86","Workstation1":1,"Workstation2":2,"Workstation3":6},
  {"Model":"HiFi87","Workstation1":3,"Workstation2":3,"Workstation3":3},
  {"Model":"HiFi88","Workstation1":8,"Workstation2":8,"Workstation3":5},
  {"Model":"HiFi89","Workstation1":3,"Workstation2":4,"Workstation3":2},
  {"Model":"HiFi90","Workstation1":3,"Workstation2":9,"Workstation3":2},
  {"Model":"HiFi91","Workstation1":3,"Workstation2":6,"Workstation3":9},
  {"Model":"HiFi92","Workstation1":3,"Workstation2":1,"Workstation3":3},
  {"Model":"HiFi93","Workstation1":6,"Workstation2":4,"Workstation3":6},
  {"Model":"HiFi94","Workstation1":7,"Workstation2":8,"Workstation3":9},
  {"Model":"HiFi95","Workstation1":6,"Workstation2":8,"Workstation3":7},
  {"Model":"HiFi96","Workstation1":2,"Workstation2":6,"Workstation3":2},
  {"Model":"HiFi97","Workstation1":1,"Workstation2":8,"Workstation3":4},
  {"Model":"HiFi98","Workstation1":8,"Workstation2":5,"Workstation3":5},
  {"Model":"HiFi99","Workstation1":9,"Workstation2":5,"Workstation3":8},
  {"Model":"HiFi100","Workstation1":7,"Workstation2":8,"Workstation3":5},
  {"Model":"HiFi101","Workstation1":9,"Workstation2":3,"Workstation3":4}
]
```

This model fully specifies the objective, constraints, variable domains, and all required data for the 101 HiFi models across the three workstations.
##### Decision Variables

$x_{ij} \ge 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

$\min \sum_{i\in I} f_i y_i + \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction: $\displaystyle \sum_{i\in I} x_{ij} = d_j , \quad \forall j\in J$  
2. Warehouse capacity: $\displaystyle \sum_{j\in J} x_{ij} \le u_i y_i , \quad \forall i\in I$  
3. Linking activation and shipment: $\displaystyle \sum_{j\in J} x_{ij} \le M y_i , \quad \forall i\in I$  
4. Domains: $x_{ij} \ge 0$ (continuous), $y_i \in \{0,1\}$.

$I=\{1,2,\dots ,11\}$, $J=\{1,2,\dots ,11\}$, $M=\sum_{j\in J} d_j = 370$.

###### Retrieved Information

{
  "warehouses": [
    {"id":1,"opening_cost":3000,"capacity":180},
    {"id":2,"opening_cost":3200,"capacity":160},
    {"id":3,"opening_cost":3100,"capacity":200},
    {"id":4,"opening_cost":2800,"capacity":150},
    {"id":5,"opening_cost":3500,"capacity":170},
    {"id":6,"opening_cost":2700,"capacity":190},
    {"id":7,"opening_cost":2900,"capacity":160},
    {"id":8,"opening_cost":3050,"capacity":175},
    {"id":9,"opening_cost":3100,"capacity":170},
    {"id":10,"opening_cost":2200,"capacity":180},
    {"id":11,"opening_cost":2890,"capacity":190}
  ],
  "stores": [
    {"id":1,"demand":30},
    {"id":2,"demand":40},
    {"id":3,"demand":20},
    {"id":4,"demand":35},
    {"id":5,"demand":20},
    {"id":6,"demand":25},
    {"id":7,"demand":45},
    {"id":8,"demand":38},
    {"id":9,"demand":32},
    {"id":10,"demand":41},
    {"id":11,"demand":44}
  ],
  "transportation_cost": [
    [12,11,14,15,17,13,12,16,16,14,15],
    [17,19,15,20,18,14,17,15,13,15,16],
    [13,14,12,14,16,15,11,14,16,18,17],
    [18,16,17,13,18,17,14,19,16,13,18],
    [10,13,12,19,15,11,12,14,12,15,17],
    [15,12,14,16,13,17,16,16,14,18,19],
    [14,13,15,17,12,13,14,15,12,16,14],
    [19,16,18,20,17,19,16,18,15,15,18],
    [17,18,12,14,16,15,14,17,21,15,18],
    [14,13,15,17,16,18,14,19,15,17,19],
    [15,13,16,17,11,13,14,15,19,21,13]
  ],
  "M":370
}
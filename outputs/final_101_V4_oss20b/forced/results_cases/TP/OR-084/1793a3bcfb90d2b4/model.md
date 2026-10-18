##### Decision Variables

$x_{ij}\in\{0,1\}$: 1 if task $j\in J$ is assigned to processor $i\in I$, 0 otherwise.  
$T\geq0$: makespan (completion time of the last task).

##### Objective Function

$\min T$

##### Constraints

1. Assignment: $\sum_{i\in I}x_{ij}=1,\quad\forall j\in J$  
2. Processor load: $\sum_{j\in J}\frac{b_j}{f_i}x_{ij}\leq T,\quad\forall i\in I$  
3. Binary: $x_{ij}\in\{0,1\},\quad\forall i\in I,\forall j\in J$  
4. Non‑negativity: $T\geq0$

$I=\{1,2,3\}$ (processor indices corresponding to frequencies 1.33, 2, 2.66 GHz)  
$J=\{1,2,\dots,40\}$ (task indices)  

$f_1=1.33,\; f_2=2,\; f_3=2.66$  

###### Retrieved Information

```json
{
  "processors": [
    {"id":"1","frequency":1.33},
    {"id":"2","frequency":2},
    {"id":"3","frequency":2.66}
  ],
  "tasks": [
    {"id":"1","BI":1.1},
    {"id":"2","BI":2.1},
    {"id":"3","BI":3},
    {"id":"4","BI":1},
    {"id":"5","BI":0.7},
    {"id":"6","BI":5},
    {"id":"7","BI":3},
    {"id":"8","BI":3.5},
    {"id":"9","BI":4.4},
    {"id":"10","BI":3.8},
    {"id":"11","BI":3.5},
    {"id":"12","BI":2.8},
    {"id":"13","BI":4.1},
    {"id":"14","BI":2.9},
    {"id":"15","BI":5.4},
    {"id":"16","BI":5.8},
    {"id":"17","BI":2.6},
    {"id":"18","BI":4.9},
    {"id":"19","BI":3.4},
    {"id":"20","BI":3.6},
    {"id":"21","BI":5.6},
    {"id":"22","BI":0.9},
    {"id":"23","BI":1},
    {"id":"24","BI":0.6},
    {"id":"25","BI":5.1},
    {"id":"26","BI":4.8},
    {"id":"27","BI":5.3},
    {"id":"28","BI":5.9},
    {"id":"29","BI":4.9},
    {"id":"30","BI":3},
    {"id":"31","BI":4.8},
    {"id":"32","BI":1.2},
    {"id":"33","BI":4},
    {"id":"34","BI":1.3},
    {"id":"35","BI":5.7},
    {"id":"36","BI":3.4},
    {"id":"37","BI":2.8},
    {"id":"38","BI":2},
    {"id":"39","BI":4.8},
    {"id":"40","BI":3}
  ]
}
```
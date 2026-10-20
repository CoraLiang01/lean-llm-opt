##### Decision Variables

$x_{ij}\in\{0,1\}$: 1 if manager $i$ is assigned to project $j$, 0 otherwise, for $i\in I$, $j\in J$.

##### Objective Function

$\displaystyle \min\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}$

##### Constraints

1. **Manager assignment**: $\displaystyle \sum_{j\in J}x_{ij}=1,\quad\forall i\in I$  
2. **Project assignment**: $\displaystyle \sum_{i\in I}x_{ij}=1,\quad\forall j\in J$  
3. **Binary domain**: $x_{ij}\in\{0,1\},\quad\forall i\in I,\;j\in J$

$ I=\{MA,MB,MC\}$ and $ J=\{P1,P2,P3\}$.  

###### Retrieved Information

```json
{
  "managers": [
    "MA",
    "MB",
    "MC"
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ],
  "cost": {
    "MA": {
      "P1": 3000,
      "P2": 3200,
      "P3": 3100
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000
    }
  }
}
```
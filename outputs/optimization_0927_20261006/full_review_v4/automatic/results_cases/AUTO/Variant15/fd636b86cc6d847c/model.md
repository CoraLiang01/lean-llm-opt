##### Sets

- Workstations: $I = \{\text{W1}, \text{W2}, \text{W3}, \text{W4}\}$
- Jobs: $J = \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}, \text{J9}\}$

##### Parameters

- Assignment cost: $c_{ij}$ is the cost of assigning job $j$ to workstation $i$.
- Resource consumption: $a_{ij}$ is the capacity consumed on workstation $i$ if job $j$ is assigned there.
- Workstation capacity: $b_i$ is the total capacity of workstation $i$.

###### Assignment Costs $c_{ij}$

|        | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|--------|----|----|----|----|----|----|----|----|----|
| **W1** |  6 |  8 | 18 | 20 | 21 | 19 | 23 | 22 | 24 |
| **W2** | 19 | 18 |  7 |  6 | 20 | 22 | 21 | 23 | 25 |
| **W3** | 22 | 21 | 20 | 19 |  5 |  7 | 18 | 20 | 21 |
| **W4** | 21 | 22 | 23 | 20 | 19 | 18 |  6 |  8 |  7 |

###### Resource Consumption $a_{ij}$

|        | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|--------|----|----|----|----|----|----|----|----|----|
| **W1** |  4 |  5 |  7 |  8 |  8 |  7 |  8 |  9 |  8 |
| **W2** |  7 |  8 |  4 |  5 |  8 |  8 |  7 |  8 |  9 |
| **W3** |  8 |  7 |  8 |  7 |  5 |  4 |  7 |  8 |  7 |
| **W4** |  8 |  8 |  9 |  8 |  7 |  7 |  4 |  5 |  4 |

###### Workstation Capacities $b_i$

- $b_{\text{W1}} = 15$
- $b_{\text{W2}} = 14$
- $b_{\text{W3}} = 16$
- $b_{\text{W4}} = 13$

##### Decision Variables

- $x_{ij} = \begin{cases} 1 & \text{if job } j \text{ is assigned to workstation } i \\ 0 & \text{otherwise} \end{cases}$

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. **Exactly-One Assignment for Each Job:**

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

2. **Workstation Capacity Constraints:**

$\sum_{j \in J} a_{ij} x_{ij} \leq b_i \quad \forall i \in I$

3. **Binary Assignment Variables:**

$x_{ij} \in \{0,1\} \quad \forall i \in I, \forall j \in J$

##### Retrieved Information

{
  "workstations": {
    "W1": {"Capacity": 15},
    "W2": {"Capacity": 14},
    "W3": {"Capacity": 16},
    "W4": {"Capacity": 13}
  },
  "jobs": [
    "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"
  ],
  "assignment_costs": {
    "W1": {"J1": 6, "J2": 8, "J3": 18, "J4": 20, "J5": 21, "J6": 19, "J7": 23, "J8": 22, "J9": 24},
    "W2": {"J1": 19, "J2": 18, "J3": 7, "J4": 6, "J5": 20, "J6": 22, "J7": 21, "J8": 23, "J9": 25},
    "W3": {"J1": 22, "J2": 21, "J3": 20, "J4": 19, "J5": 5, "J6": 7, "J7": 18, "J8": 20, "J9": 21},
    "W4": {"J1": 21, "J2": 22, "J3": 23, "J4": 20, "J5": 19, "J6": 18, "J7": 6, "J8": 8, "J9": 7}
  },
  "assignment_resources": {
    "W1": {"J1": 4, "J2": 5, "J3": 7, "J4": 8, "J5": 8, "J6": 7, "J7": 8, "J8": 9, "J9": 8},
    "W2": {"J1": 7, "J2": 8, "J3": 4, "J4": 5, "J5": 8, "J6": 8, "J7": 7, "J8": 8, "J9": 9},
    "W3": {"J1": 8, "J2": 7, "J3": 8, "J4": 7, "J5": 5, "J6": 4, "J7": 7, "J8": 8, "J9": 7},
    "W4": {"J1": 8, "J2": 8, "J3": 9, "J4": 8, "J5": 7, "J6": 7, "J7": 4, "J8": 5, "J9": 4}
  }
}
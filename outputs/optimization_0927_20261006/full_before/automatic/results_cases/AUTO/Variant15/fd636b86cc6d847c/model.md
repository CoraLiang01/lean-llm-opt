##### Objective Function:

$\quad \min \sum_{i \in \{W1, W2, W3, W4\}} \sum_{j \in \{J1, J2, ..., J9\}} c_{ij} x_{ij}$

where $x_{ij} = 1$ if job $j$ is assigned to workstation $i$, $0$ otherwise, and $c_{ij}$ is the assignment cost.

##### Constraints

###### 1. Exactly-One Assignment Constraints (Each job assigned to one workstation):

$\sum_{i \in \{W1, W2, W3, W4\}} x_{ij} = 1 \quad \forall j \in \{J1, J2, ..., J9\}$

###### 2. Capacity Constraints (Workstation capacity not exceeded):

$\sum_{j \in \{J1, J2, ..., J9\}} a_{ij} x_{ij} \leq b_i \quad \forall i \in \{W1, W2, W3, W4\}$

where $a_{ij}$ is the capacity consumed by assigning job $j$ to workstation $i$, and $b_i$ is the capacity of workstation $i$.

###### 3. Binary Assignment Variables:

$x_{ij} \in \{0,1\} \quad \forall i \in \{W1, W2, W3, W4\},\ j \in \{J1, J2, ..., J9\}$

---

##### Retrieved Information

**Workstations and Capacities:**

| Workstation | Capacity ($b_i$) |
|-------------|------------------|
| W1          | 15               |
| W2          | 14               |
| W3          | 16               |
| W4          | 13               |

**Jobs:** $J1, J2, J3, J4, J5, J6, J7, J8, J9$

**Assignment Costs ($c_{ij}$):**

|             | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|-------------|----|----|----|----|----|----|----|----|----|
| **W1**      | 6  | 8  | 18 | 20 | 21 | 19 | 23 | 22 | 24 |
| **W2**      | 19 | 18 | 7  | 6  | 20 | 22 | 21 | 23 | 25 |
| **W3**      | 22 | 21 | 20 | 19 | 5  | 7  | 18 | 20 | 21 |
| **W4**      | 21 | 22 | 23 | 20 | 19 | 18 | 6  | 8  | 7  |

**Assignment Resource Consumption ($a_{ij}$):**

|             | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|-------------|----|----|----|----|----|----|----|----|----|
| **W1**      | 4  | 5  | 7  | 8  | 8  | 7  | 8  | 9  | 8  |
| **W2**      | 7  | 8  | 4  | 5  | 8  | 8  | 7  | 8  | 9  |
| **W3**      | 8  | 7  | 8  | 7  | 5  | 4  | 7  | 8  | 7  |
| **W4**      | 8  | 8  | 9  | 8  | 7  | 7  | 4  | 5  | 4  |

**Variables:**

$x_{ij} \in \{0,1\}$ for all $i \in \{W1, W2, W3, W4\}$ and $j \in \{J1, J2, ..., J9\}$

---

**Full Model:**

$\displaystyle \min \sum_{i \in \{W1, W2, W3, W4\}} \sum_{j \in \{J1, J2, ..., J9\}} c_{ij} x_{ij}$

Subject to:

$\sum_{i \in \{W1, W2, W3, W4\}} x_{ij} = 1 \quad \forall j \in \{J1, J2, ..., J9\}$

$\sum_{j \in \{J1, J2, ..., J9\}} a_{ij} x_{ij} \leq b_i \quad \forall i \in \{W1, W2, W3, W4\}$

$x_{ij} \in \{0,1\} \quad \forall i, j$
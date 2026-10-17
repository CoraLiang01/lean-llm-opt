##### Objective Function:

$\quad \min \sum_{i \in \{\text{W1}, \text{W2}, \text{W3}, \text{W4}\}} \sum_{j \in \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}, \text{J9}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Exactly-One Assignment Constraints (Each job assigned to exactly one workstation):

$\sum_{i \in \{\text{W1}, \text{W2}, \text{W3}, \text{W4}\}} x_{ij} = 1 \quad \forall j \in \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}, \text{J9}\}$

###### 2. Capacity Constraints (Total resource consumption on each workstation does not exceed its capacity):

$\sum_{j \in \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}, \text{J9}\}} r_{ij} x_{ij} \leq \text{Cap}_i \quad \forall i \in \{\text{W1}, \text{W2}, \text{W3}, \text{W4}\}$

###### 3. Binary Assignment Variables:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{W1}, \text{W2}, \text{W3}, \text{W4}\},\; j \in \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}, \text{J9}\}$

---

##### Retrieved Information

**Workstations and Capacities:**
- W1: 15
- W2: 14
- W3: 16
- W4: 13

**Jobs:**
- J1, J2, J3, J4, J5, J6, J7, J8, J9

**Assignment Costs $c_{ij}$:**

|        | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|--------|----|----|----|----|----|----|----|----|----|
| **W1** |  6 |  8 | 18 | 20 | 21 | 19 | 23 | 22 | 24 |
| **W2** | 19 | 18 |  7 |  6 | 20 | 22 | 21 | 23 | 25 |
| **W3** | 22 | 21 | 20 | 19 |  5 |  7 | 18 | 20 | 21 |
| **W4** | 21 | 22 | 23 | 20 | 19 | 18 |  6 |  8 |  7 |

**Assignment Resource Consumption $r_{ij}$:**

|        | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 |
|--------|----|----|----|----|----|----|----|----|----|
| **W1** |  4 |  5 |  7 |  8 |  8 |  7 |  8 |  9 |  8 |
| **W2** |  7 |  8 |  4 |  5 |  8 |  8 |  7 |  8 |  9 |
| **W3** |  8 |  7 |  8 |  7 |  5 |  4 |  7 |  8 |  7 |
| **W4** |  8 |  8 |  9 |  8 |  7 |  7 |  4 |  5 |  4 |

**Variable Definitions:**
- $x_{ij} = 1$ if job $j$ is assigned to workstation $i$, $0$ otherwise.

**Parameter Definitions:**
- $c_{ij}$: Assignment cost for assigning job $j$ to workstation $i$ (see table above).
- $r_{ij}$: Resource consumption for assigning job $j$ to workstation $i$ (see table above).
- $\text{Cap}_i$: Capacity of workstation $i$ (see list above).
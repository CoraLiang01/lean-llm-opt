##### Sets and Indices

- Let $W = \{1,2,\ldots,12\}$ be the set of workers, indexed by $i$.
- Let $T = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$ be the set of tasks, indexed by $j$.

##### Parameters

Let $c_{ij}$ denote the time required for worker $i$ to complete task $j$, as given below:

| Worker |  A |  B |  C |  D |  E |  F |  G |  H |  I |  J |
|--------|----|----|----|----|----|----|----|----|----|----|
|   1    |  9 |  4 |  3 |  7 |  6 |  5 |  6 |  3 |  7 |  5 |
|   2    |  4 |  6 |  5 |  6 |  4 |  5 |  3 |  8 |  7 |  6 |
|   3    |  5 |  4 |  7 |  5 |  6 |  6 |  5 |  8 |  6 |  9 |
|   4    |  7 |  5 |  2 |  3 |  7 |  8 |  5 |  6 |  8 |  5 |
|   5    | 10 |  6 |  7 |  4 |  5 |  4 |  4 |  5 |  9 |  7 |
|   6    |  6 |  7 |  6 |  3 |  9 |  5 |  7 |  4 |  3 |  4 |
|   7    |  8 |  8 |  5 |  9 |  5 |  7 |  5 |  9 |  5 |  3 |
|   8    |  7 |  4 |  8 |  8 |  6 |  7 |  5 |  7 |  7 |  7 |
|   9    |  5 |  6 |  8 |  7 |  7 |  8 |  7 |  8 |  4 |  5 |
|  10    |  8 |  7 |  9 |  5 |  8 |  5 |  9 |  9 |  3 |  4 |
|  11    |  9 |  8 | 10 |  8 |  5 |  4 |  7 |  6 |  8 |  7 |
|  12    |  8 |  5 |  6 |  9 |  4 |  7 |  8 |  4 |  7 |  9 |

##### Decision Variables

- $x_{ij} \in \{0,1\}$: 1 if worker $i$ is assigned to task $j$, 0 otherwise.
- $y_i \in \{0,1\}$: 1 if worker $i$ is selected (assigned to any task), 0 otherwise.

##### Objective Function

$\min \sum_{i=1}^{12} \sum_{j \in T} c_{ij} x_{ij}$

##### Constraints

1. **Each task is assigned to exactly one worker:**

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in T$

2. **Each selected worker is assigned to exactly one task, and unselected workers are assigned to none:**

$\sum_{j \in T} x_{ij} = y_i \quad \forall i \in W$

3. **Exactly 10 workers are selected:**

$\sum_{i=1}^{12} y_i = 10$

4. **Variable domains:**

$x_{ij} \in \{0,1\} \quad \forall i \in W, \forall j \in T$

$y_i \in \{0,1\} \quad \forall i \in W$

##### Retrieved Information

{
  "workers": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
  "tasks": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
  "time": {
    "1":  {"A":9,  "B":4,  "C":3,  "D":7,  "E":6,  "F":5,  "G":6,  "H":3,  "I":7,  "J":5},
    "2":  {"A":4,  "B":6,  "C":5,  "D":6,  "E":4,  "F":5,  "G":3,  "H":8,  "I":7,  "J":6},
    "3":  {"A":5,  "B":4,  "C":7,  "D":5,  "E":6,  "F":6,  "G":5,  "H":8,  "I":6,  "J":9},
    "4":  {"A":7,  "B":5,  "C":2,  "D":3,  "E":7,  "F":8,  "G":5,  "H":6,  "I":8,  "J":5},
    "5":  {"A":10, "B":6,  "C":7,  "D":4,  "E":5,  "F":4,  "G":4,  "H":5,  "I":9,  "J":7},
    "6":  {"A":6,  "B":7,  "C":6,  "D":3,  "E":9,  "F":5,  "G":7,  "H":4,  "I":3,  "J":4},
    "7":  {"A":8,  "B":8,  "C":5,  "D":9,  "E":5,  "F":7,  "G":5,  "H":9,  "I":5,  "J":3},
    "8":  {"A":7,  "B":4,  "C":8,  "D":8,  "E":6,  "F":7,  "G":5,  "H":7,  "I":7,  "J":7},
    "9":  {"A":5,  "B":6,  "C":8,  "D":7,  "E":7,  "F":8,  "G":7,  "H":8,  "I":4,  "J":5},
    "10": {"A":8,  "B":7,  "C":9,  "D":5,  "E":8,  "F":5,  "G":9,  "H":9,  "I":3,  "J":4},
    "11": {"A":9,  "B":8,  "C":10, "D":8,  "E":5,  "F":4,  "G":7,  "H":6,  "I":8,  "J":7},
    "12": {"A":8,  "B":5,  "C":6,  "D":9,  "E":4,  "F":7,  "G":8,  "H":4,  "I":7,  "J":9}
  }
}
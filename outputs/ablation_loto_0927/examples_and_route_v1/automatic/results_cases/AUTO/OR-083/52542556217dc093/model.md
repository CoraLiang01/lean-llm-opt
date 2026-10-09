##### Decision Variables

Let $x_{ij}$ be a binary variable:
- $x_{ij} = 1$ if worker $i$ is assigned to task $j$,
- $x_{ij} = 0$ otherwise,

where $i \in W$ (the set of 12 workers) and $j \in T$ (the set of 12 tasks).

##### Parameters

Let $t_{ij}$ denote the time required for worker $i$ to complete task $j$, as given in the table below.

###### Workers:
$W = \{A, B, C, D, E, F, G, H, I, J\}$

###### Tasks:
$T = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\}$

###### Time Matrix ($t_{ij}$):

| Task |  A |  B |  C |  D |  E |  F |  G |  H |  I |  J |
|------|----|----|----|----|----|----|----|----|----|----|
|  1   |  9 |  4 |  3 |  7 |  6 |  5 |  6 |  3 |  7 |  5 |
|  2   |  4 |  6 |  5 |  6 |  4 |  5 |  3 |  8 |  7 |  6 |
|  3   |  5 |  4 |  7 |  5 |  6 |  6 |  5 |  8 |  6 |  9 |
|  4   |  7 |  5 |  2 |  3 |  7 |  8 |  5 |  6 |  8 |  5 |
|  5   | 10 |  6 |  7 |  4 |  5 |  4 |  4 |  5 |  9 |  7 |
|  6   |  6 |  7 |  6 |  3 |  9 |  5 |  7 |  4 |  3 |  4 |
|  7   |  8 |  8 |  5 |  9 |  5 |  7 |  5 |  9 |  5 |  3 |
|  8   |  7 |  4 |  8 |  8 |  6 |  7 |  5 |  7 |  7 |  7 |
|  9   |  5 |  6 |  8 |  7 |  7 |  8 |  7 |  8 |  4 |  5 |
| 10   |  8 |  7 |  9 |  5 |  8 |  5 |  9 |  9 |  3 |  4 |
| 11   |  9 |  8 | 10 |  8 |  5 |  4 |  7 |  6 |  8 |  7 |
| 12   |  8 |  5 |  6 |  9 |  4 |  7 |  8 |  4 |  7 |  9 |

##### Objective Function

$\min \sum_{i \in W} \sum_{j \in T} t_{ij} x_{ij}$

##### Constraints

1. **Each task is assigned to exactly one worker:**

$\sum_{i \in W} x_{ij} = 1 \quad \forall j \in S$

where $S \subset T$, $|S| = 10$ (the selected 10 tasks).

2. **Each selected worker is assigned to exactly one task:**

$\sum_{j \in S} x_{ij} \leq 1 \quad \forall i \in W$

3. **Exactly 10 workers are assigned:**

$\sum_{i \in W} \sum_{j \in S} x_{ij} = 10$

4. **Each worker is assigned to at most one task:**

$\sum_{j \in S} x_{ij} \leq 1 \quad \forall i \in W$

5. **Each task is assigned to exactly one worker:**

$\sum_{i \in W} x_{ij} = 1 \quad \forall j \in S$

6. **Binary variables:**

$x_{ij} \in \{0,1\} \quad \forall i \in W, \forall j \in S$

##### Retrieved Information

{
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
  "tasks": [1,2,3,4,5,6,7,8,9,10,11,12],
  "time": {
    "A": [9,4,5,7,10,6,8,7,5,8,9,8],
    "B": [4,6,4,5,6,7,8,4,6,7,8,5],
    "C": [3,5,7,2,7,6,5,8,8,9,10,6],
    "D": [7,6,5,3,4,3,9,8,7,5,8,9],
    "E": [6,4,6,7,5,9,5,6,7,8,5,4],
    "F": [5,5,6,8,4,5,7,7,8,5,4,7],
    "G": [6,3,5,5,4,7,5,5,7,9,7,8],
    "H": [3,8,8,6,5,4,9,7,8,9,6,4],
    "I": [7,7,6,8,9,3,5,7,4,3,8,7],
    "J": [5,6,9,5,7,4,3,7,5,4,7,9]
  }
}

##### Notes

- The model selects 10 out of 12 workers and assigns each to exactly one of 10 tasks (out of 12), with each task assigned to exactly one worker, minimizing total working hours.
- If the set of tasks to be completed is fixed (e.g., tasks 1–10), then $S = \{1,2,3,4,5,6,7,8,9,10\}$. If any 10 tasks can be chosen, introduce additional selection variables for tasks.
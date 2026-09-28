##### Decision Variables

Let $x_{wt}$ be a binary variable:

$$
x_{wt} = 
\begin{cases}
1 & \text{if worker } w \text{ is assigned to task } t \\
0 & \text{otherwise}
\end{cases}
$$

where $w \in \{A, B, C, D, E, F, G, H, I, J, K, L\}$ and $t \in \{1,2,3,4,5,6,7,8,9,10\}$.

##### Parameters

Let $c_{wt}$ denote the time required for worker $w$ to complete task $t$, as given below:

| Worker | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|----|
| A | 9 | 4 | 3 | 7 | 6 | 5 | 6 | 3 | 7 | 5 |
| B | 4 | 6 | 5 | 6 | 4 | 5 | 3 | 8 | 7 | 6 |
| C | 5 | 4 | 7 | 5 | 6 | 6 | 5 | 8 | 6 | 9 |
| D | 7 | 5 | 2 | 3 | 7 | 8 | 5 | 6 | 8 | 5 |
| E | 10 | 6 | 7 | 4 | 5 | 4 | 4 | 5 | 9 | 7 |
| F | 6 | 7 | 6 | 3 | 9 | 5 | 7 | 4 | 3 | 4 |
| G | 8 | 8 | 5 | 9 | 5 | 7 | 5 | 9 | 5 | 3 |
| H | 7 | 4 | 8 | 8 | 6 | 7 | 5 | 7 | 7 | 7 |
| I | 5 | 6 | 8 | 7 | 7 | 8 | 7 | 8 | 4 | 5 |
| J | 8 | 7 | 9 | 5 | 8 | 5 | 9 | 9 | 3 | 4 |
| K | 9 | 8 | 10 | 8 | 5 | 4 | 7 | 6 | 8 | 7 |
| L | 8 | 5 | 6 | 9 | 4 | 7 | 8 | 4 | 7 | 9 |

##### Objective Function

$$
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
$$

where $W = \{A, B, C, D, E, F, G, H, I, J, K, L\}$ and $T = \{1,2,3,4,5,6,7,8,9,10\}$.

##### Constraints

1. **Each task is assigned to exactly one worker:**

$$
\sum_{w \in W} x_{wt} = 1 \quad \forall t \in T
$$

2. **Each worker is assigned to at most one task:**

$$
\sum_{t \in T} x_{wt} \leq 1 \quad \forall w \in W
$$

3. **Exactly 10 workers are assigned (i.e., 2 workers are not assigned any task):**

$$
\sum_{w \in W} \sum_{t \in T} x_{wt} = 10
$$

4. **Variable domain:**

$$
x_{wt} \in \{0,1\} \quad \forall w \in W, \forall t \in T
$$

##### Retrieved Information

{
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
  "tasks": [1,2,3,4,5,6,7,8,9,10],
  "time": {
    "A": [9, 4, 3, 7, 6, 5, 6, 3, 7, 5],
    "B": [4, 6, 5, 6, 4, 5, 3, 8, 7, 6],
    "C": [5, 4, 7, 5, 6, 6, 5, 8, 6, 9],
    "D": [7, 5, 2, 3, 7, 8, 5, 6, 8, 5],
    "E": [10, 6, 7, 4, 5, 4, 4, 5, 9, 7],
    "F": [6, 7, 6, 3, 9, 5, 7, 4, 3, 4],
    "G": [8, 8, 5, 9, 5, 7, 5, 9, 5, 3],
    "H": [7, 4, 8, 8, 6, 7, 5, 7, 7, 7],
    "I": [5, 6, 8, 7, 7, 8, 7, 8, 4, 5],
    "J": [8, 7, 9, 5, 8, 5, 9, 9, 3, 4],
    "K": [9, 8, 10, 8, 5, 4, 7, 6, 8, 7],
    "L": [8, 5, 6, 9, 4, 7, 8, 4, 7, 9]
  }
}
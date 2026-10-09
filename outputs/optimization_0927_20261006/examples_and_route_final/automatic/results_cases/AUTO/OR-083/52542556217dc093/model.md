##### Sets and Indices

- Let $W = \{A, B, C, D, E, F, G, H, I, J, K, L\}$ be the set of 12 workers. (From the data, the columns are A, B, C, D, E, F, G, H, I, J; the remaining two workers can be denoted as K, L if present in the full data. Here, only 10 workers are listed, so we proceed with $W = \{A, B, C, D, E, F, G, H, I, J\}$ and clarify that only 10 workers are available.)
- Let $T = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of 10 tasks.

##### Parameters

Let $c_{wt}$ be the time required for worker $w$ to complete task $t$, as given in the table below:

| Task |   A | B | C | D | E | F | G | H | I | J |
|------|----|---|---|---|---|---|---|---|---|---|
| 1    |  9 | 4 | 3 | 7 | 6 | 5 | 6 | 3 | 7 | 5 |
| 2    |  4 | 6 | 5 | 6 | 4 | 5 | 3 | 8 | 7 | 6 |
| 3    |  5 | 4 | 7 | 5 | 6 | 6 | 5 | 8 | 6 | 9 |
| 4    |  7 | 5 | 2 | 3 | 7 | 8 | 5 | 6 | 8 | 5 |
| 5    | 10 | 6 | 7 | 4 | 5 | 4 | 4 | 5 | 9 | 7 |
| 6    |  6 | 7 | 6 | 3 | 9 | 5 | 7 | 4 | 3 | 4 |
| 7    |  8 | 8 | 5 | 9 | 5 | 7 | 5 | 9 | 5 | 3 |
| 8    |  7 | 4 | 8 | 8 | 6 | 7 | 5 | 7 | 7 | 7 |
| 9    |  5 | 6 | 8 | 7 | 7 | 8 | 7 | 8 | 4 | 5 |
| 10   |  8 | 7 | 9 | 5 | 8 | 5 | 9 | 9 | 3 | 4 |

##### Decision Variables

- $x_{wt} = \begin{cases} 1 & \text{if worker } w \text{ is assigned to task } t \\ 0 & \text{otherwise} \end{cases}$

##### Objective Function

Minimize the total working hours:
$$
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
$$

##### Constraints

1. **Each task is assigned to exactly one worker:**
   $$
   \sum_{w \in W} x_{wt} = 1 \quad \forall t \in T
   $$

2. **Each worker is assigned to at most one task:**
   $$
   \sum_{t \in T} x_{wt} \leq 1 \quad \forall w \in W
   $$

3. **Exactly 10 workers are assigned (since there are 10 tasks):**
   $$
   \sum_{w \in W} \sum_{t \in T} x_{wt} = 10
   $$

   (This is automatically satisfied by the above two constraints, but can be included for clarity.)

4. **Variable domains:**
   $$
   x_{wt} \in \{0,1\} \quad \forall w \in W, \forall t \in T
   $$

##### Retrieved Information

{
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
  "tasks": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "time": {
    "A": {"1": 9, "2": 4, "3": 5, "4": 7, "5": 10, "6": 6, "7": 8, "8": 7, "9": 5, "10": 8},
    "B": {"1": 4, "2": 6, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 4, "9": 6, "10": 7},
    "C": {"1": 3, "2": 5, "3": 7, "4": 2, "5": 7, "6": 6, "7": 5, "8": 8, "9": 8, "10": 9},
    "D": {"1": 7, "2": 6, "3": 5, "4": 3, "5": 4, "6": 3, "7": 9, "8": 8, "9": 7, "10": 5},
    "E": {"1": 6, "2": 4, "3": 6, "4": 7, "5": 5, "6": 9, "7": 5, "8": 6, "9": 7, "10": 8},
    "F": {"1": 5, "2": 5, "3": 6, "4": 8, "5": 4, "6": 5, "7": 7, "8": 7, "9": 8, "10": 5},
    "G": {"1": 6, "2": 3, "3": 5, "4": 5, "5": 4, "6": 7, "7": 5, "8": 5, "9": 7, "10": 9},
    "H": {"1": 3, "2": 8, "3": 8, "4": 6, "5": 5, "6": 4, "7": 9, "8": 7, "9": 8, "10": 9},
    "I": {"1": 7, "2": 7, "3": 6, "4": 8, "5": 9, "6": 3, "7": 5, "8": 7, "9": 4, "10": 3},
    "J": {"1": 5, "2": 6, "3": 9, "4": 5, "5": 7, "6": 4, "7": 3, "8": 7, "9": 5, "10": 4}
  }
}

If there are actually 12 workers in the full data, extend $W$ and $c_{wt}$ accordingly, and the model remains the same.
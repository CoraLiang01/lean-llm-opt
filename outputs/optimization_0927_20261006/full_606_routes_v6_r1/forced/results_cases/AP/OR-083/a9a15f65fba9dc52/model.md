##### Sets and Indices

- Let $W = \{A, B, C, D, E, F, G, H, I, J, K, L\}$ be the set of 12 workers (here, K and L are implied as the 11th and 12th workers, but only A–J are present in the data; if K and L exist, their data would be needed).
- Let $T = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of 10 tasks.

##### Parameters

Let $c_{w,t}$ be the time required for worker $w$ to complete task $t$, as given below:

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

- $x_{w,t} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise.
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise.

##### Objective Function

$$
\min \sum_{w \in W} \sum_{t \in T} c_{w,t} \cdot x_{w,t}
$$

##### Constraints

1. **Each task is assigned to exactly one worker:**

$$
\sum_{w \in W} x_{w,t} = 1 \quad \forall t \in T
$$

2. **Each selected worker is assigned to exactly one task (and unselected workers are assigned to none):**

$$
\sum_{t \in T} x_{w,t} = y_w \quad \forall w \in W
$$

3. **Exactly 10 workers are selected:**

$$
\sum_{w \in W} y_w = 10
$$

4. **Variables are binary:**

$$
x_{w,t} \in \{0,1\} \quad \forall w \in W, \forall t \in T
$$

$$
y_w \in \{0,1\} \quad \forall w \in W
$$

##### Retrieved Information

{
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
  },
  "tasks": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
}

**Note:** If there are additional workers (K, L), their time data must be included similarly. The model above is based on the 10 workers (A–J) present in the data. If the full set of 12 workers is available, extend $W$ and $c_{w,t}$ accordingly.
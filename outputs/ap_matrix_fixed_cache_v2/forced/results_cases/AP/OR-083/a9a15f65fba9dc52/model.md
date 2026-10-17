##### Sets

- Let $W = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}, \text{K}, \text{L}\}$ be the set of 12 workers. (From the data, only A–J are present; if only 10 workers are available, use those. If 12, extend as needed.)
- Let $T = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of 10 tasks.

##### Parameters

Let $t_{w,t}$ be the time required for worker $w$ to complete task $t$, as given below:

| Worker/Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------------|---|---|---|---|---|---|---|---|---|----|----|----|
| A           | 9 | 4 | 5 | 7 |10 | 6 | 8 | 7 | 5 | 8  | 9  | 8  |
| B           | 4 | 6 | 4 | 5 | 6 | 7 | 8 | 4 | 6 | 7  | 8  | 5  |
| C           | 3 | 5 | 7 | 2 | 7 | 6 | 5 | 8 | 8 | 9  |10  | 6  |
| D           | 7 | 6 | 5 | 3 | 4 | 3 | 9 | 8 | 7 | 5  | 8  | 9  |
| E           | 6 | 4 | 6 | 7 | 5 | 9 | 5 | 6 | 7 | 8  | 5  | 4  |
| F           | 5 | 5 | 6 | 8 | 4 | 5 | 7 | 7 | 8 | 5  | 4  | 7  |
| G           | 6 | 3 | 5 | 5 | 4 | 7 | 5 | 5 | 7 | 9  | 7  | 8  |
| H           | 3 | 8 | 8 | 6 | 5 | 4 | 9 | 7 | 8 | 9  | 6  | 4  |
| I           | 7 | 7 | 6 | 8 | 9 | 3 | 5 | 7 | 4 | 3  | 8  | 7  |
| J           | 5 | 6 | 9 | 5 | 7 | 4 | 3 | 7 | 5 | 4  | 7  | 9  |

(If there are only 10 workers, use A–J; if 12, extend as needed.)

##### Decision Variables

- $x_{w,t} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise.
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise.

##### Objective Function

$\min \sum_{w \in W} \sum_{t \in T} t_{w,t} \cdot x_{w,t}$

##### Constraints

1. **Each task is assigned to exactly one worker:**

$\sum_{w \in W} x_{w,t} = 1 \quad \forall t \in T$

2. **Each selected worker is assigned to at most one task:**

$\sum_{t \in T} x_{w,t} \leq y_w \quad \forall w \in W$

3. **Exactly 10 workers are selected:**

$\sum_{w \in W} y_w = 10$

4. **Each worker is assigned to at most one task:**

$\sum_{t \in T} x_{w,t} \leq 1 \quad \forall w \in W$

5. **Variable domains:**

$x_{w,t} \in \{0,1\} \quad \forall w \in W, t \in T$

$y_w \in \{0,1\} \quad \forall w \in W$

##### Retrieved Information

{
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
  "tasks": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "time": {
    "A": {"1": 9, "2": 4, "3": 5, "4": 7, "5": 10, "6": 6, "7": 8, "8": 7, "9": 5, "10": 8, "11": 9, "12": 8},
    "B": {"1": 4, "2": 6, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 4, "9": 6, "10": 7, "11": 8, "12": 5},
    "C": {"1": 3, "2": 5, "3": 7, "4": 2, "5": 7, "6": 6, "7": 5, "8": 8, "9": 8, "10": 9, "11": 10, "12": 6},
    "D": {"1": 7, "2": 6, "3": 5, "4": 3, "5": 4, "6": 3, "7": 9, "8": 8, "9": 7, "10": 5, "11": 8, "12": 9},
    "E": {"1": 6, "2": 4, "3": 6, "4": 7, "5": 5, "6": 9, "7": 5, "8": 6, "9": 7, "10": 8, "11": 5, "12": 4},
    "F": {"1": 5, "2": 5, "3": 6, "4": 8, "5": 4, "6": 5, "7": 7, "8": 7, "9": 8, "10": 5, "11": 4, "12": 7},
    "G": {"1": 6, "2": 3, "3": 5, "4": 5, "5": 4, "6": 7, "7": 5, "8": 5, "9": 7, "10": 9, "11": 7, "12": 8},
    "H": {"1": 3, "2": 8, "3": 8, "4": 6, "5": 5, "6": 4, "7": 9, "8": 7, "9": 8, "10": 9, "11": 6, "12": 4},
    "I": {"1": 7, "2": 7, "3": 6, "4": 8, "5": 9, "6": 3, "7": 5, "8": 7, "9": 4, "10": 3, "11": 8, "12": 7},
    "J": {"1": 5, "2": 6, "3": 9, "4": 5, "5": 7, "6": 4, "7": 3, "8": 7, "9": 5, "10": 4, "11": 7, "12": 9}
  }
}

- All time values are as given in the table above for each worker-task pair.
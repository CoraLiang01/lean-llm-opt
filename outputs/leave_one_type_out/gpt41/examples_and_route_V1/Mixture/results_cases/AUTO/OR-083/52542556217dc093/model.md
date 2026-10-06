##### Decision Variables:

Let $x_{ij}$ be a binary variable:
- $x_{ij} = 1$ if worker $i$ is assigned to task $j$
- $x_{ij} = 0$ otherwise

where $i \in \{1,2,\ldots,12\}$ (workers), $j \in \{\text{A},\text{B},\ldots,\text{J}\}$ (tasks).

##### Parameters:

Let $t_{ij}$ denote the time required for worker $i$ to complete task $j$, as given below:

| Worker $i$ | $t_{iA}$ | $t_{iB}$ | $t_{iC}$ | $t_{iD}$ | $t_{iE}$ | $t_{iF}$ | $t_{iG}$ | $t_{iH}$ | $t_{iI}$ | $t_{iJ}$ |
|:----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1          | 9        | 4        | 3        | 7        | 6        | 5        | 6        | 3        | 7        | 5        |
| 2          | 4        | 6        | 5        | 6        | 4        | 5        | 3        | 8        | 7        | 6        |
| 3          | 5        | 4        | 7        | 5        | 6        | 6        | 5        | 8        | 6        | 9        |
| 4          | 7        | 5        | 2        | 3        | 7        | 8        | 5        | 6        | 8        | 5        |
| 5          | 10       | 6        | 7        | 4        | 5        | 4        | 4        | 5        | 9        | 7        |
| 6          | 6        | 7        | 6        | 3        | 9        | 5        | 7        | 4        | 3        | 4        |
| 7          | 8        | 8        | 5        | 9        | 5        | 7        | 5        | 9        | 5        | 3        |
| 8          | 7        | 4        | 8        | 8        | 6        | 7        | 5        | 7        | 7        | 7        |
| 9          | 5        | 6        | 8        | 7        | 7        | 8        | 7        | 8        | 4        | 5        |
| 10         | 8        | 7        | 9        | 5        | 8        | 5        | 9        | 9        | 3        | 4        |
| 11         | 9        | 8        | 10       | 8        | 5        | 4        | 7        | 6        | 8        | 7        |
| 12         | 8        | 5        | 6        | 9        | 4        | 7        | 8        | 4        | 7        | 9        |

##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{10} t_{ij} x_{ij}$

##### Constraints:

1. **Each task is assigned to exactly one worker:**

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{\text{A},\text{B},\ldots,\text{J}\}$

2. **Each worker is assigned to at most one task:**

$\sum_{j=1}^{10} x_{ij} \leq 1 \quad \forall i \in \{1,2,\ldots,12\}$

3. **Exactly 10 workers are selected and assigned to tasks:**

$\sum_{i=1}^{12} \sum_{j=1}^{10} x_{ij} = 10$

(Alternatively, since each task must be assigned to one worker and there are 10 tasks, this is automatically satisfied by the first constraint.)

4. **Variable domain:**

$x_{ij} \in \{0,1\} \quad \forall i \in \{1,2,\ldots,12\},\ j \in \{\text{A},\text{B},\ldots,\text{J}\}$

##### Retrieved Information

{
  "time": {
    "1": {"A": 9, "B": 4, "C": 3, "D": 7, "E": 6, "F": 5, "G": 6, "H": 3, "I": 7, "J": 5},
    "2": {"A": 4, "B": 6, "C": 5, "D": 6, "E": 4, "F": 5, "G": 3, "H": 8, "I": 7, "J": 6},
    "3": {"A": 5, "B": 4, "C": 7, "D": 5, "E": 6, "F": 6, "G": 5, "H": 8, "I": 6, "J": 9},
    "4": {"A": 7, "B": 5, "C": 2, "D": 3, "E": 7, "F": 8, "G": 5, "H": 6, "I": 8, "J": 5},
    "5": {"A": 10, "B": 6, "C": 7, "D": 4, "E": 5, "F": 4, "G": 4, "H": 5, "I": 9, "J": 7},
    "6": {"A": 6, "B": 7, "C": 6, "D": 3, "E": 9, "F": 5, "G": 7, "H": 4, "I": 3, "J": 4},
    "7": {"A": 8, "B": 8, "C": 5, "D": 9, "E": 5, "F": 7, "G": 5, "H": 9, "I": 5, "J": 3},
    "8": {"A": 7, "B": 4, "C": 8, "D": 8, "E": 6, "F": 7, "G": 5, "H": 7, "I": 7, "J": 7},
    "9": {"A": 5, "B": 6, "C": 8, "D": 7, "E": 7, "F": 8, "G": 7, "H": 8, "I": 4, "J": 5},
    "10": {"A": 8, "B": 7, "C": 9, "D": 5, "E": 8, "F": 5, "G": 9, "H": 9, "I": 3, "J": 4},
    "11": {"A": 9, "B": 8, "C": 10, "D": 8, "E": 5, "F": 4, "G": 7, "H": 6, "I": 8, "J": 7},
    "12": {"A": 8, "B": 5, "C": 6, "D": 9, "E": 4, "F": 7, "G": 8, "H": 4, "I": 7, "J": 9}
  },
  "workers": [1,2,3,4,5,6,7,8,9,10,11,12],
  "tasks": ["A","B","C","D","E","F","G","H","I","J"]
}
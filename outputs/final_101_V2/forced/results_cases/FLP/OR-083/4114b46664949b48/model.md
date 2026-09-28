##### Decision Variables

Let $x_{ij} \in \{0,1\}$:  
$x_{ij} = 1$ if worker $i$ is assigned to task $j$, $0$ otherwise, for $i \in W$, $j \in T$.

Let $y_i \in \{0,1\}$:  
$y_i = 1$ if worker $i$ is selected (assigned to any task), $0$ otherwise, for $i \in W$.

##### Parameters

- $W = \{1,2,\ldots,12\}$ (worker IDs)
- $T = \{\text{A},\text{B},\text{C},\text{D},\text{E},\text{F},\text{G},\text{H},\text{I},\text{J}\}$ (task IDs)
- $c_{ij}$: time required for worker $i$ to complete task $j$, as given below:

| $i$ | $c_{iA}$ | $c_{iB}$ | $c_{iC}$ | $c_{iD}$ | $c_{iE}$ | $c_{iF}$ | $c_{iG}$ | $c_{iH}$ | $c_{iI}$ | $c_{iJ}$ |
|----:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1   | 9        | 4        | 3        | 7        | 6        | 5        | 6        | 3        | 7        | 5        |
| 2   | 4        | 6        | 5        | 6        | 4        | 5        | 3        | 8        | 7        | 6        |
| 3   | 5        | 4        | 7        | 5        | 6        | 6        | 5        | 8        | 6        | 9        |
| 4   | 7        | 5        | 2        | 3        | 7        | 8        | 5        | 6        | 8        | 5        |
| 5   | 10       | 6        | 7        | 4        | 5        | 4        | 4        | 5        | 9        | 7        |
| 6   | 6        | 7        | 6        | 3        | 9        | 5        | 7        | 4        | 3        | 4        |
| 7   | 8        | 8        | 5        | 9        | 5        | 7        | 5        | 9        | 5        | 3        |
| 8   | 7        | 4        | 8        | 8        | 6        | 7        | 5        | 7        | 7        | 7        |
| 9   | 5        | 6        | 8        | 7        | 7        | 8        | 7        | 8        | 4        | 5        |
| 10  | 8        | 7        | 9        | 5        | 8        | 5        | 9        | 9        | 3        | 4        |
| 11  | 9        | 8        | 10       | 8        | 5        | 4        | 7        | 6        | 8        | 7        |
| 12  | 8        | 5        | 6        | 9        | 4        | 7        | 8        | 4        | 7        | 9        |

##### Objective Function

\[
\min \sum_{i \in W} \sum_{j \in T} c_{ij} x_{ij}
\]

##### Constraints

1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{i \in W} x_{ij} = 1, \quad \forall j \in T
   \]

2. **Each selected worker is assigned to exactly one task (and unselected workers are assigned to none):**
   \[
   \sum_{j \in T} x_{ij} = y_i, \quad \forall i \in W
   \]

3. **Exactly 10 workers are selected:**
   \[
   \sum_{i \in W} y_i = 10
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in W,\, j \in T
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in W
   \]

##### Summary of Sets and Parameters

- $W = \{1,2,3,4,5,6,7,8,9,10,11,12\}$
- $T = \{\text{A},\text{B},\text{C},\text{D},\text{E},\text{F},\text{G},\text{H},\text{I},\text{J}\}$
- $c_{ij}$ as given in the table above.

This model selects 10 out of 12 workers and assigns each to exactly one task, with each task assigned to exactly one worker, minimizing the total working hours.
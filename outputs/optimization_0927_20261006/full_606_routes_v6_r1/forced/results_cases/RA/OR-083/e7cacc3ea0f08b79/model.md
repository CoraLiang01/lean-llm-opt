Let $W = \{1,2,\ldots,12\}$ be the set of workers, and $T = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$ be the set of tasks.

Let $c_{it}$ denote the time required for worker $i$ to complete task $t$, as given in the table below.

Define binary variables:
- $x_{it} = \begin{cases} 1 & \text{if worker } i \text{ is assigned to task } t \\ 0 & \text{otherwise} \end{cases}$
- $y_i = \begin{cases} 1 & \text{if worker } i \text{ is selected (assigned to any task)} \\ 0 & \text{otherwise} \end{cases}$

Objective:
\[
\min \sum_{i \in W} \sum_{t \in T} c_{it} x_{it}
\]

Subject to:

1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{i \in W} x_{it} = 1 \quad \forall t \in T
   \]

2. **Each selected worker is assigned to exactly one task, and unselected workers are assigned to none:**
   \[
   \sum_{t \in T} x_{it} = y_i \quad \forall i \in W
   \]

3. **Exactly 10 workers are selected:**
   \[
   \sum_{i \in W} y_i = 10
   \]

4. **Variable domains:**
   \[
   x_{it} \in \{0,1\} \quad \forall i \in W,\, t \in T
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in W
   \]

---

#### Data Table ($c_{it}$: time required for worker $i$ to complete task $t$):

| Worker |   A | B | C | D | E | F | G | H | I | J |
|--------|----|---|---|---|---|---|---|---|---|---|
| 1      |  9 | 4 | 3 | 7 | 6 | 5 | 6 | 3 | 7 | 5 |
| 2      |  4 | 6 | 5 | 6 | 4 | 5 | 3 | 8 | 7 | 6 |
| 3      |  5 | 4 | 7 | 5 | 6 | 6 | 5 | 8 | 6 | 9 |
| 4      |  7 | 5 | 2 | 3 | 7 | 8 | 5 | 6 | 8 | 5 |
| 5      | 10 | 6 | 7 | 4 | 5 | 4 | 4 | 5 | 9 | 7 |
| 6      |  6 | 7 | 6 | 3 | 9 | 5 | 7 | 4 | 3 | 4 |
| 7      |  8 | 8 | 5 | 9 | 5 | 7 | 5 | 9 | 5 | 3 |
| 8      |  7 | 4 | 8 | 8 | 6 | 7 | 5 | 7 | 7 | 7 |
| 9      |  5 | 6 | 8 | 7 | 7 | 8 | 7 | 8 | 4 | 5 |
| 10     |  8 | 7 | 9 | 5 | 8 | 5 | 9 | 9 | 3 | 4 |
| 11     |  9 | 8 |10 | 8 | 5 | 4 | 7 | 6 | 8 | 7 |
| 12     |  8 | 5 | 6 | 9 | 4 | 7 | 8 | 4 | 7 | 9 |

All indices and coefficients are as in the table above.
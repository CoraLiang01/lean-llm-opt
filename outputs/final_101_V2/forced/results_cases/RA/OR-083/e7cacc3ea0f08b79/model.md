Let $W = \{1,2,\ldots,12\}$ be the set of workers, and $T = \{\text{A}, \text{B}, \ldots, \text{J}\}$ be the set of tasks.

Let $c_{wt}$ be the time required for worker $w$ to complete task $t$, as given in the table below.

Let $x_{wt}$ be a binary variable:
\[
x_{wt} = 
\begin{cases}
1 & \text{if worker } w \text{ is assigned to task } t \\
0 & \text{otherwise}
\end{cases}
\]

Let $y_w$ be a binary variable:
\[
y_w = 
\begin{cases}
1 & \text{if worker } w \text{ is selected (assigned to any task)} \\
0 & \text{otherwise}
\end{cases}
\]

#### Parameters (from 15.csv):

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

#### Mathematical Model

Minimize total working hours:
\[
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
\]

Subject to:

1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{w \in W} x_{wt} = 1 \quad \forall t \in T
   \]

2. **Each selected worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{wt} \leq y_w \quad \forall w \in W
   \]

3. **Exactly 10 workers are selected:**
   \[
   \sum_{w \in W} y_w = 10
   \]

4. **Each selected worker is assigned to exactly one task (since there are 10 tasks and 10 workers):**
   \[
   \sum_{t \in T} x_{wt} \geq y_w \quad \forall w \in W
   \]
   (Alternatively, since $x_{wt}$ can only be 1 if $y_w=1$, and each task must be assigned, this is implied.)

5. **Variable domains:**
   \[
   x_{wt} \in \{0,1\} \quad \forall w \in W,\, t \in T
   \]
   \[
   y_w \in \{0,1\} \quad \forall w \in W
   \]

#### Where

- $W = \{1,2,\ldots,12\}$ (workers)
- $T = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$ (tasks)
- $c_{wt}$ is the time required for worker $w$ to complete task $t$, as given in the table above.

This model selects 10 out of 12 workers and assigns each to exactly one task, minimizing the total working hours.
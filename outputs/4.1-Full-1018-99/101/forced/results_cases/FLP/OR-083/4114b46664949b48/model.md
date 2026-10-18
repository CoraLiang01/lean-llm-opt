##### Sets

- $W = \{1,2,\ldots,12\}$: set of workers.
- $T = \{A,B,C,D,E,F,G,H,I,J\}$: set of tasks.

##### Parameters

Let $c_{wt}$ be the time required for worker $w$ to complete task $t$, as given below:

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

##### Decision Variables

- $x_{wt} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise.
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise.

##### Objective Function

\[
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
\]

##### Constraints

1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{w \in W} x_{wt} = 1, \quad \forall t \in T
   \]

2. **Each worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{wt} \leq y_w, \quad \forall w \in W
   \]

3. **Exactly 10 workers are selected:**
   \[
   \sum_{w \in W} y_w = 10
   \]

4. **Assignment only if worker is selected:**
   \[
   x_{wt} \leq y_w, \quad \forall w \in W, \forall t \in T
   \]

5. **Variable domains:**
   \[
   x_{wt} \in \{0,1\}, \quad y_w \in \{0,1\}
   \]

##### Parameters (full matrix):

\[
\begin{array}{c|cccccccccc}
\text{Worker} & A & B & C & D & E & F & G & H & I & J \\
\hline
1 & 9 & 4 & 3 & 7 & 6 & 5 & 6 & 3 & 7 & 5 \\
2 & 4 & 6 & 5 & 6 & 4 & 5 & 3 & 8 & 7 & 6 \\
3 & 5 & 4 & 7 & 5 & 6 & 6 & 5 & 8 & 6 & 9 \\
4 & 7 & 5 & 2 & 3 & 7 & 8 & 5 & 6 & 8 & 5 \\
5 & 10 & 6 & 7 & 4 & 5 & 4 & 4 & 5 & 9 & 7 \\
6 & 6 & 7 & 6 & 3 & 9 & 5 & 7 & 4 & 3 & 4 \\
7 & 8 & 8 & 5 & 9 & 5 & 7 & 5 & 9 & 5 & 3 \\
8 & 7 & 4 & 8 & 8 & 6 & 7 & 5 & 7 & 7 & 7 \\
9 & 5 & 6 & 8 & 7 & 7 & 8 & 7 & 8 & 4 & 5 \\
10 & 8 & 7 & 9 & 5 & 8 & 5 & 9 & 9 & 3 & 4 \\
11 & 9 & 8 & 10 & 8 & 5 & 4 & 7 & 6 & 8 & 7 \\
12 & 8 & 5 & 6 & 9 & 4 & 7 & 8 & 4 & 7 & 9 \\
\end{array}
\]

##### Summary

- Assign exactly 10 out of 12 workers to 10 tasks (one task per selected worker, one worker per task), minimizing total working hours according to the provided time matrix.
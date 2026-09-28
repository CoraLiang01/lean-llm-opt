Let:
- $T = \{1,2,\ldots,40\}$ be the set of tasks (indexed by $i$).
- $P = \{1,2,3\}$ be the set of CPUs (indexed by $j$), with frequencies $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (in GHz).
- $b_i$ be the number of billions of instructions (BI) for task $i$ (from the BI row in 18.csv).
- $x_{ij} \in \{0,1\}$: $x_{ij} = 1$ if task $i$ is assigned to CPU $j$, $0$ otherwise.
- $C_{\max}$: the makespan (completion time of the last task).

Parameters (from 18.csv, in source order):

| Task $i$ | $b_i$ |
|----------|-------|
| 1        | 1.1   |
| 2        | 2.1   |
| 3        | 3     |
| 4        | 1     |
| 5        | 0.7   |
| 6        | 5     |
| 7        | 3     |
| 8        | 3.5   |
| 9        | 4.4   |
| 10       | 3.8   |
| 11       | 3.5   |
| 12       | 2.8   |
| 13       | 4.1   |
| 14       | 2.9   |
| 15       | 5.4   |
| 16       | 5.8   |
| 17       | 2.6   |
| 18       | 4.9   |
| 19       | 3.4   |
| 20       | 3.6   |
| 21       | 5.6   |
| 22       | 0.9   |
| 23       | 1     |
| 24       | 0.6   |
| 25       | 5.1   |
| 26       | 4.8   |
| 27       | 5.3   |
| 28       | 5.9   |
| 29       | 4.9   |
| 30       | 3     |
| 31       | 4.8   |
| 32       | 1.2   |
| 33       | 4     |
| 34       | 1.3   |
| 35       | 5.7   |
| 36       | 3.4   |
| 37       | 2.8   |
| 38       | 2     |
| 39       | 4.8   |
| 40       | 3     |

Formulation:

Minimize the makespan:
$$
\min C_{\max}
$$

Subject to:

1. **Assignment constraints:** Each task is assigned to exactly one CPU:
$$
\sum_{j=1}^3 x_{ij} = 1 \quad \forall i \in T
$$

2. **Makespan constraints:** For each CPU, the total processing time of its assigned tasks does not exceed $C_{\max}$:
$$
\sum_{i=1}^{40} \frac{b_i}{f_j} x_{ij} \leq C_{\max} \quad \forall j \in P
$$

3. **Variable domains:**
$$
x_{ij} \in \{0,1\} \quad \forall i \in T,\, j \in P
$$
$$
C_{\max} \geq 0
$$

Where:
- $b_i$ is the BI for task $i$ (see table above).
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (GHz).

All data and indices are preserved in source order.
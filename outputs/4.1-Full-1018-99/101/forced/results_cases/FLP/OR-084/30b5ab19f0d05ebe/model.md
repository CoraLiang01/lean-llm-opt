##### Parameters

- Number of tasks: $N = 40$
- Number of CPUs: $M = 3$
- CPU frequencies (in GHz): $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- Task instruction counts (in billions of instructions, BI):

| Task $j$ | $a_j$ (BI) |
|----------|------------|
| 1        | 1.1        |
| 2        | 2.1        |
| 3        | 3          |
| 4        | 1          |
| 5        | 0.7        |
| 6        | 5          |
| 7        | 3          |
| 8        | 3.5        |
| 9        | 4.4        |
| 10       | 3.8        |
| 11       | 3.5        |
| 12       | 2.8        |
| 13       | 4.1        |
| 14       | 2.9        |
| 15       | 5.4        |
| 16       | 5.8        |
| 17       | 2.6        |
| 18       | 4.9        |
| 19       | 3.4        |
| 20       | 3.6        |
| 21       | 5.6        |
| 22       | 0.9        |
| 23       | 1          |
| 24       | 0.6        |
| 25       | 5.1        |
| 26       | 4.8        |
| 27       | 5.3        |
| 28       | 5.9        |
| 29       | 4.9        |
| 30       | 3          |
| 31       | 4.8        |
| 32       | 1.2        |
| 33       | 4          |
| 34       | 1.3        |
| 35       | 5.7        |
| 36       | 3.4        |
| 37       | 2.8        |
| 38       | 2          |
| 39       | 4.8        |
| 40       | 3          |

##### Decision Variables

- $x_{ij} \in \{0,1\}$: $1$ if task $j$ is assigned to CPU $i$, $0$ otherwise, for $i=1,2,3$, $j=1,\ldots,40$.
- $C_{\max} \geq 0$: the makespan (completion time of the last finishing CPU).

##### Derived Parameters

- Processing time of task $j$ on CPU $i$: $p_{ij} = \dfrac{a_j}{f_i}$

##### Objective

$\min C_{\max}$

##### Constraints

1. **Each task assigned to exactly one CPU:**
   $$
   \sum_{i=1}^3 x_{ij} = 1, \quad \forall j = 1,\ldots,40
   $$

2. **Makespan lower bounds CPU total load:**
   $$
   \sum_{j=1}^{40} p_{ij} x_{ij} \leq C_{\max}, \quad \forall i = 1,2,3
   $$

3. **Variable domains:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i=1,2,3,\ j=1,\ldots,40
   $$
   $$
   C_{\max} \geq 0
   $$

##### Full Model

Let $a_j$ be the instruction count for task $j$ (see table above), $f_i$ the frequency of CPU $i$.

Minimize:
$$
C_{\max}
$$

Subject to:
$$
\sum_{i=1}^3 x_{ij} = 1, \quad \forall j = 1,\ldots,40
$$
$$
\sum_{j=1}^{40} \frac{a_j}{f_i} x_{ij} \leq C_{\max}, \quad \forall i = 1,2,3
$$
$$
x_{ij} \in \{0,1\}, \quad \forall i=1,2,3,\ j=1,\ldots,40
$$
$$
C_{\max} \geq 0
$$

Where:

- $a_j$ as listed above for $j=1,\ldots,40$
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
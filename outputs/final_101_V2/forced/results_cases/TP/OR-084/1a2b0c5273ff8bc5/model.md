##### Sets and Parameters

- Let $T = \{1,2,\ldots,40\}$ be the set of tasks.
- Let $P = \{1,2,3\}$ be the set of CPUs.
- Let $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (GHz) be the frequencies of CPUs 1, 2, and 3.
- Let $b_t$ be the number of billions of instructions (BI) for task $t$:

| Task $t$ | $b_t$ |
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

##### Decision Variables

- $x_{tp} \in \{0,1\}$: $=1$ if task $t$ is assigned to CPU $p$, $0$ otherwise, for all $t \in T$, $p \in P$.
- $C_{\max} \geq 0$: the makespan (completion time of the last task).

##### Objective

$\min C_{\max}$

##### Constraints

1. **Assignment:** Each task is assigned to exactly one CPU:
   $$
   \sum_{p=1}^3 x_{tp} = 1 \quad \forall t \in T
   $$

2. **Makespan:** For each CPU, the total processing time of its assigned tasks does not exceed $C_{\max}$:
   $$
   \sum_{t=1}^{40} \frac{b_t}{f_p} x_{tp} \leq C_{\max} \quad \forall p \in \{1,2,3\}
   $$
   where $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$.

3. **Variable domains:**
   $$
   x_{tp} \in \{0,1\} \quad \forall t \in T,\, p \in P
   $$
   $$
   C_{\max} \geq 0
   $$

##### Complete Model

$\displaystyle
\begin{align*}
\min\ & C_{\max} \\
\text{s.t.}\quad
& \sum_{p=1}^3 x_{tp} = 1 && \forall t = 1,\ldots,40 \\
& \sum_{t=1}^{40} \frac{b_t}{f_p} x_{tp} \leq C_{\max} && \forall p = 1,2,3 \\
& x_{tp} \in \{0,1\} && \forall t = 1,\ldots,40;\ p=1,2,3 \\
& C_{\max} \geq 0
\end{align*}
$

where $b_t$ is as listed above, and $(f_1, f_2, f_3) = (1.33, 2, 2.66)$.
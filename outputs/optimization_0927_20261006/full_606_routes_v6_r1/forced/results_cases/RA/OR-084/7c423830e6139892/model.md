Let:
- $T = \{1,2,\ldots,40\}$ be the set of tasks.
- $P = \{1,2,3\}$ be the set of CPUs.
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (GHz) be the frequencies of CPUs 1, 2, and 3.
- $b_t$ be the number of billions of instructions (BI) for task $t$ (from the data below).
- $x_{tp} \in \{0,1\}$: 1 if task $t$ is assigned to processor $p$, 0 otherwise.
- $s_t \geq 0$: start time of task $t$.
- $C_t \geq 0$: completion time of task $t$.
- $C_{\max} \geq 0$: makespan (completion time of the last task).

Parameters (from 18.csv):

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

Let $f_p$ be the frequency of processor $p$ as above.

Define $proc\_time_{tp} = \dfrac{b_t}{f_p}$: processing time of task $t$ on processor $p$.

#### Decision Variables

- $x_{tp} \in \{0,1\}$, $\forall t \in T, p \in P$
- $s_t \geq 0$, $\forall t \in T$
- $C_t \geq 0$, $\forall t \in T$
- $C_{\max} \geq 0$

#### Objective

$$
\min C_{\max}
$$

#### Constraints

1. **Each task assigned to exactly one processor:**
   $$
   \sum_{p \in P} x_{tp} = 1, \quad \forall t \in T
   $$

2. **Completion time definition:**
   $$
   C_t = s_t + \sum_{p \in P} \left( \frac{b_t}{f_p} \cdot x_{tp} \right), \quad \forall t \in T
   $$

3. **Sequencing constraints (no overlap on each processor):**

   For all $t, t' \in T$, $t \neq t'$, $p \in P$:

   Introduce binary variables $y_{tt'p} \in \{0,1\}$, where $y_{tt'p} = 1$ if task $t$ precedes $t'$ on processor $p$.

   For all $t \neq t'$, $p$:
   $$
   s_{t'} \geq C_t - M \cdot (1 - y_{tt'p}) - M \cdot (1 - x_{tp}) - M \cdot (1 - x_{t'p}), \quad \forall t \neq t', p
   $$
   $$
   s_t \geq C_{t'} - M \cdot y_{tt'p} - M \cdot (1 - x_{tp}) - M \cdot (1 - x_{t'p}), \quad \forall t \neq t', p
   $$
   $$
   y_{tt'p} + y_{t'tp} = 1, \quad \forall t \neq t', p
   $$
   (Here, $M$ is a sufficiently large constant.)

4. **Makespan definition:**
   $$
   C_{\max} \geq C_t, \quad \forall t \in T
   $$

5. **Variable domains:**
   $$
   x_{tp} \in \{0,1\}, \quad \forall t \in T, p \in P
   $$
   $$
   y_{tt'p} \in \{0,1\}, \quad \forall t \neq t', p \in P
   $$
   $$
   s_t \geq 0, \quad C_t \geq 0, \quad C_{\max} \geq 0
   $$

#### Data

- $T = \{1,2,\ldots,40\}$
- $P = \{1,2,3\}$
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- $b_t$ as listed above

This model assigns each task to exactly one processor, sequences the tasks on each processor, and minimizes the makespan.
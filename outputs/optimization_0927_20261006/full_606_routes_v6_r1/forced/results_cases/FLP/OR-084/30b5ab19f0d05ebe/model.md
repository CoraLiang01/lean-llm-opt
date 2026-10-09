##### Parameters

- Let $T = \{1,2,\ldots,40\}$ be the set of tasks.
- Let $P = \{1,2,3\}$ be the set of CPUs.
- CPU frequencies: $f_1 = 1.33$ GHz, $f_2 = 2$ GHz, $f_3 = 2.66$ GHz.
- Task instruction counts (in billions of instructions, BI):

| Task $j$ | $I_j$ (BI) |
|----------|-----------|
| 1        | 1.1       |
| 2        | 2.1       |
| 3        | 3         |
| 4        | 1         |
| 5        | 0.7       |
| 6        | 5         |
| 7        | 3         |
| 8        | 3.5       |
| 9        | 4.4       |
| 10       | 3.8       |
| 11       | 3.5       |
| 12       | 2.8       |
| 13       | 4.1       |
| 14       | 2.9       |
| 15       | 5.4       |
| 16       | 5.8       |
| 17       | 2.6       |
| 18       | 4.9       |
| 19       | 3.4       |
| 20       | 3.6       |
| 21       | 5.6       |
| 22       | 0.9       |
| 23       | 1         |
| 24       | 0.6       |
| 25       | 5.1       |
| 26       | 4.8       |
| 27       | 5.3       |
| 28       | 5.9       |
| 29       | 4.9       |
| 30       | 3         |
| 31       | 4.8       |
| 32       | 1.2       |
| 33       | 4         |
| 34       | 1.3       |
| 35       | 5.7       |
| 36       | 3.4       |
| 37       | 2.8       |
| 38       | 2         |
| 39       | 4.8       |
| 40       | 3         |

##### Decision Variables

- $x_{jp} \in \{0,1\}$: $1$ if task $j$ is assigned to CPU $p$, $0$ otherwise, for $j \in T$, $p \in P$.
- $C_{\max} \geq 0$: the makespan (completion time of the last task).

##### Derived Parameters

- Processing time of task $j$ on CPU $p$: $t_{jp} = \dfrac{I_j}{f_p}$

##### Objective

\[
\min C_{\max}
\]

##### Constraints

1. **Each task assigned to exactly one CPU:**
   \[
   \sum_{p \in P} x_{jp} = 1 \qquad \forall j \in T
   \]

2. **Makespan lower bounds CPU total load:**
   \[
   \sum_{j \in T} t_{jp} x_{jp} \leq C_{\max} \qquad \forall p \in P
   \]

3. **Variable domains:**
   \[
   x_{jp} \in \{0,1\} \qquad \forall j \in T,\, p \in P
   \]
   \[
   C_{\max} \geq 0
   \]

##### Explicit Data

- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- $I_j$ as listed above for $j=1,\ldots,40$
- $t_{jp} = I_j / f_p$ for all $j,p$

##### Full Model

\[
\begin{align*}
\min\quad & C_{\max} \\
\text{s.t.}\quad
& \sum_{p=1}^3 x_{jp} = 1 && \forall j=1,\ldots,40 \\
& \sum_{j=1}^{40} \frac{I_j}{f_p} x_{jp} \leq C_{\max} && \forall p=1,2,3 \\
& x_{jp} \in \{0,1\} && \forall j=1,\ldots,40;\ p=1,2,3 \\
& C_{\max} \geq 0
\end{align*}
\]

Where $I_j$ is as listed above, and $f_1=1.33$, $f_2=2$, $f_3=2.66$.
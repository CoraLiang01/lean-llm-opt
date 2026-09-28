##### Parameters

- Let $T = \{1,2,\ldots,40\}$ be the set of tasks.
- Let $P = \{1,2,3\}$ be the set of CPUs.
- CPU frequencies: $f_1 = 1.33$ GHz, $f_2 = 2$ GHz, $f_3 = 2.66$ GHz.
- Let $b_t$ be the number of basic instructions (in billions) for task $t \in T$:

\[
\begin{align*}
&b_1 = 1.1,\quad b_2 = 2.1,\quad b_3 = 3,\quad b_4 = 1,\quad b_5 = 0.7,\\
&b_6 = 5,\quad b_7 = 3,\quad b_8 = 3.5,\quad b_9 = 4.4,\quad b_{10} = 3.8,\\
&b_{11} = 3.5,\quad b_{12} = 2.8,\quad b_{13} = 4.1,\quad b_{14} = 2.9,\quad b_{15} = 5.4,\\
&b_{16} = 5.8,\quad b_{17} = 2.6,\quad b_{18} = 4.9,\quad b_{19} = 3.4,\quad b_{20} = 3.6,\\
&b_{21} = 5.6,\quad b_{22} = 0.9,\quad b_{23} = 1,\quad b_{24} = 0.6,\quad b_{25} = 5.1,\\
&b_{26} = 4.8,\quad b_{27} = 5.3,\quad b_{28} = 5.9,\quad b_{29} = 4.9,\quad b_{30} = 3,\\
&b_{31} = 4.8,\quad b_{32} = 1.2,\quad b_{33} = 4,\quad b_{34} = 1.3,\quad b_{35} = 5.7,\\
&b_{36} = 3.4,\quad b_{37} = 2.8,\quad b_{38} = 2,\quad b_{39} = 4.8,\quad b_{40} = 3
\end{align*}
\]

- Processing time of task $t$ on CPU $p$: $p_{tp} = \dfrac{b_t}{f_p}$

##### Decision Variables

- $x_{tp} \in \{0,1\}$: $1$ if task $t$ is assigned to CPU $p$, $0$ otherwise.
- $C_{\max} \geq 0$: the makespan (completion time of the last task).

##### Objective

\[
\min C_{\max}
\]

##### Constraints

1. **Each task is assigned to exactly one CPU:**
   \[
   \sum_{p \in P} x_{tp} = 1 \qquad \forall t \in T
   \]

2. **Makespan bounds the total processing time on each CPU:**
   \[
   \sum_{t \in T} p_{tp} x_{tp} \leq C_{\max} \qquad \forall p \in P
   \]
   where $p_{tp} = \dfrac{b_t}{f_p}$.

3. **Variable domains:**
   \[
   x_{tp} \in \{0,1\} \qquad \forall t \in T,\, p \in P
   \]
   \[
   C_{\max} \geq 0
   \]

##### Explicit Data

- $T = \{1,2,\ldots,40\}$
- $P = \{1,2,3\}$
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- $b_t$ as listed above
- $p_{tp}$ matrix: $p_{tp} = b_t / f_p$ for all $t,p$

##### Full Model

\[
\begin{align*}
\min\ & C_{\max} \\
\text{s.t.}\quad
& \sum_{p=1}^3 x_{tp} = 1 \qquad \forall t = 1,\ldots,40 \\
& \sum_{t=1}^{40} \frac{b_t}{f_p} x_{tp} \leq C_{\max} \qquad \forall p = 1,2,3 \\
& x_{tp} \in \{0,1\} \qquad \forall t = 1,\ldots,40;\ p = 1,2,3 \\
& C_{\max} \geq 0
\end{align*}
\]

where $b_t$ and $f_p$ are as specified above.
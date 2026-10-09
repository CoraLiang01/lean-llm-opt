##### Sets and Parameters

- Let $T = \{1,2,\ldots,40\}$ be the set of tasks.
- Let $P = \{1,2,3\}$ be the set of CPUs.
- Let $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (GHz) be the frequencies of CPUs 1, 2, and 3.
- Let $b_t$ be the number of billions of instructions (BI) for task $t$:

\[
\begin{array}{ll}
b_1 = 1.1 & b_{21} = 5.6 \\
b_2 = 2.1 & b_{22} = 0.9 \\
b_3 = 3 & b_{23} = 1 \\
b_4 = 1 & b_{24} = 0.6 \\
b_5 = 0.7 & b_{25} = 5.1 \\
b_6 = 5 & b_{26} = 4.8 \\
b_7 = 3 & b_{27} = 5.3 \\
b_8 = 3.5 & b_{28} = 5.9 \\
b_9 = 4.4 & b_{29} = 4.9 \\
b_{10} = 3.8 & b_{30} = 3 \\
b_{11} = 3.5 & b_{31} = 4.8 \\
b_{12} = 2.8 & b_{32} = 1.2 \\
b_{13} = 4.1 & b_{33} = 4 \\
b_{14} = 2.9 & b_{34} = 1.3 \\
b_{15} = 5.4 & b_{35} = 5.7 \\
b_{16} = 5.8 & b_{36} = 3.4 \\
b_{17} = 2.6 & b_{37} = 2.8 \\
b_{18} = 4.9 & b_{38} = 2 \\
b_{19} = 3.4 & b_{39} = 4.8 \\
b_{20} = 3.6 & b_{40} = 3 \\
\end{array}
\]

##### Decision Variables

- $x_{tp} \in \{0,1\}$: $=1$ if task $t$ is assigned to CPU $p$, $0$ otherwise, for all $t \in T$, $p \in P$.
- $C_{\max} \geq 0$: makespan (completion time of the last task).

##### Objective

\[
\min C_{\max}
\]

##### Constraints

1. **Each task assigned to exactly one CPU:**
   \[
   \sum_{p=1}^3 x_{tp} = 1 \qquad \forall t \in T
   \]

2. **Makespan lower bounds for each CPU:**
   \[
   \sum_{t=1}^{40} \frac{b_t}{f_p} x_{tp} \leq C_{\max} \qquad \forall p \in \{1,2,3\}
   \]

3. **Variable domains:**
   \[
   x_{tp} \in \{0,1\} \qquad \forall t \in T,\, p \in P
   \]
   \[
   C_{\max} \geq 0
   \]

##### Data

- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- $b_t$ as listed above for $t=1,\ldots,40$

##### Model Summary

Assign each task to exactly one CPU. The processing time of task $t$ on CPU $p$ is $b_t / f_p$. The total load on each CPU must not exceed $C_{\max}$. Minimize $C_{\max}$.
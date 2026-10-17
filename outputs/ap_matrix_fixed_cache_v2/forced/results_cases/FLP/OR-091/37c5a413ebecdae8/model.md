##### Decision Variables

Let $x_i \in \{0,1\}$ for each Operations Research course $i$, where $x_i = 1$ if course $i$ is selected, $0$ otherwise.

##### Parameters

Let the set of eligible courses be:
- $C22$: Operations Research: Linear Programming, credits $=5$, interest points $=95$
- $C23$: Integer Programming, credits $=5$, interest points $=92$
- $C24$: Stochastic Processes, credits $=4$, interest points $=86$
- $C25$: Simulation Modeling, credits $=4$, interest points $=82$
- $C26$: Network Flows, credits $=4$, interest points $=85$
- $C27$: Queueing Theory, credits $=4$, interest points $=80$
- $C28$: Revenue Management, credits $=4$, interest points $=88$

Let $I = \{C22, C23, C24, C25, C26, C27, C28\}$.

Let $c_i$ be the number of credits for course $i$:

\[
\begin{align*}
c_{C22} &= 5 \\
c_{C23} &= 5 \\
c_{C24} &= 4 \\
c_{C25} &= 4 \\
c_{C26} &= 4 \\
c_{C27} &= 4 \\
c_{C28} &= 4 \\
\end{align*}
\]

Let $p_i$ be the interest points for course $i$:

\[
\begin{align*}
p_{C22} &= 95 \\
p_{C23} &= 92 \\
p_{C24} &= 86 \\
p_{C25} &= 82 \\
p_{C26} &= 85 \\
p_{C27} &= 80 \\
p_{C28} &= 88 \\
\end{align*}
\]

##### Objective Function

\[
\max \sum_{i \in I} p_i x_i = 95x_{C22} + 92x_{C23} + 86x_{C24} + 82x_{C25} + 85x_{C26} + 80x_{C27} + 88x_{C28}
\]

##### Constraints

1. Credit limit:
   \[
   \sum_{i \in I} c_i x_i \leq 20
   \]
   That is,
   \[
   5x_{C22} + 5x_{C23} + 4x_{C24} + 4x_{C25} + 4x_{C26} + 4x_{C27} + 4x_{C28} \leq 20
   \]

2. Binary selection:
   \[
   x_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

Eligible Operations Research courses:

| course_id | course_name                              | credits | interest_points |
|-----------|------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming  | 5       | 95              |
| C23       | Integer Programming                      | 5       | 92              |
| C24       | Stochastic Processes                     | 4       | 86              |
| C25       | Simulation Modeling                      | 4       | 82              |
| C26       | Network Flows                            | 4       | 85              |
| C27       | Queueing Theory                          | 4       | 80              |
| C28       | Revenue Management                       | 4       | 88              |
Let $T = \{1,2,\ldots,40\}$ be the set of tasks, and $P = \{1,2,3\}$ the set of CPUs. Let $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$ (in GHz) be the frequencies of CPUs 1, 2, and 3, respectively. Let $b_t$ be the number of billions of instructions (BI) for task $t$, as given below.

Define binary variables:
\[
x_{tp} = 
\begin{cases}
1 & \text{if task } t \text{ is assigned to CPU } p \\
0 & \text{otherwise}
\end{cases}
\]
Let $C_t$ be the completion time of task $t$ (in seconds), and $C_{\max}$ the makespan.

Parameters (from 18.csv):

\[
\begin{array}{ll}
\text{Task} & b_t \\
1 & 1.1 \\
2 & 2.1 \\
3 & 3 \\
4 & 1 \\
5 & 0.7 \\
6 & 5 \\
7 & 3 \\
8 & 3.5 \\
9 & 4.4 \\
10 & 3.8 \\
11 & 3.5 \\
12 & 2.8 \\
13 & 4.1 \\
14 & 2.9 \\
15 & 5.4 \\
16 & 5.8 \\
17 & 2.6 \\
18 & 4.9 \\
19 & 3.4 \\
20 & 3.6 \\
21 & 5.6 \\
22 & 0.9 \\
23 & 1 \\
24 & 0.6 \\
25 & 5.1 \\
26 & 4.8 \\
27 & 5.3 \\
28 & 5.9 \\
29 & 4.9 \\
30 & 3 \\
31 & 4.8 \\
32 & 1.2 \\
33 & 4 \\
34 & 1.3 \\
35 & 5.7 \\
36 & 3.4 \\
37 & 2.8 \\
38 & 2 \\
39 & 4.8 \\
40 & 3 \\
\end{array}
\]

Processing time of task $t$ on CPU $p$:
\[
\text{proc}_{tp} = \frac{b_t}{f_p}
\]

#### Objective:
\[
\min C_{\max}
\]

#### Constraints:

1. **Assignment:** Each task is assigned to exactly one CPU:
\[
\sum_{p=1}^3 x_{tp} = 1 \qquad \forall t \in T
\]

2. **Sequencing and Completion Times:** For each CPU $p$, let $S_{t}$ be the start time of task $t$. For any two distinct tasks $t, t'$, define binary variables $y_{tt'p}$, where $y_{tt'p} = 1$ if task $t$ precedes $t'$ on CPU $p$.

For all $t \neq t'$, $p$:
\[
S_{t'} \geq S_t + \frac{b_t}{f_p} - M \cdot (1 - x_{tp} - x_{t'p} + y_{tt'p})
\]
\[
S_t \geq S_{t'} + \frac{b_{t'}}{f_p} - M \cdot (1 - x_{tp} - x_{t'p} + 1 - y_{tt'p})
\]
where $M$ is a sufficiently large constant.

Alternatively, a compact disjunctive formulation can be used, but the above is explicit.

3. **Completion time definition:**
\[
C_t = S_t + \sum_{p=1}^3 x_{tp} \cdot \frac{b_t}{f_p} \qquad \forall t \in T
\]

4. **Makespan:**
\[
C_{\max} \geq C_t \qquad \forall t \in T
\]

5. **Variable domains:**
\[
x_{tp} \in \{0,1\} \qquad \forall t \in T,\, p \in P
\]
\[
y_{tt'p} \in \{0,1\} \qquad \forall t \neq t',\, p \in P
\]
\[
S_t \geq 0 \qquad \forall t \in T
\]
\[
C_t \geq 0 \qquad \forall t \in T
\]
\[
C_{\max} \geq 0
\]

#### Summary of Parameters:

- $T = \{1,2,\ldots,40\}$
- $P = \{1,2,3\}$
- $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$
- $b_t$ as listed above

#### Decision Variables:

- $x_{tp}$: binary, 1 if task $t$ assigned to CPU $p$
- $y_{tt'p}$: binary, 1 if $t$ precedes $t'$ on $p$
- $S_t$: start time of task $t$
- $C_t$: completion time of task $t$
- $C_{\max}$: makespan

#### Objective:

\[
\min C_{\max}
\]

#### Subject to:

\[
\sum_{p=1}^3 x_{tp} = 1 \qquad \forall t
\]
\[
S_{t'} \geq S_t + \frac{b_t}{f_p} - M \cdot (1 - x_{tp} - x_{t'p} + y_{tt'p}) \qquad \forall t \neq t',\, p
\]
\[
S_t \geq S_{t'} + \frac{b_{t'}}{f_p} - M \cdot (1 - x_{tp} - x_{t'p} + 1 - y_{tt'p}) \qquad \forall t \neq t',\, p
\]
\[
C_t = S_t + \sum_{p=1}^3 x_{tp} \cdot \frac{b_t}{f_p} \qquad \forall t
\]
\[
C_{\max} \geq C_t \qquad \forall t
\]
\[
x_{tp} \in \{0,1\} \qquad \forall t,p
\]
\[
y_{tt'p} \in \{0,1\} \qquad \forall t \neq t',\, p
\]
\[
S_t \geq 0,\, C_t \geq 0,\, C_{\max} \geq 0
\]

All parameters and variables are as defined above, with $b_t$ values taken directly from the provided 18.csv.
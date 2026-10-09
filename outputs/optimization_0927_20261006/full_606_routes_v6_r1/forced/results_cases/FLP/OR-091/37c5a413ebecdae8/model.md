##### Decision Variables

Let $x_i \in \{0,1\}$ for each Operations Research course $i$, where $x_i = 1$ if course $i$ is selected, $0$ otherwise.

##### Parameters

Let the set of Operations Research courses be:
- $I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$

For each course $i \in I$:
- $c_i$: number of credits for course $i$
- $p_i$: interest points for course $i$

The data retrieved is:

| course_id | course_name                              | credits | interest_points |
|-----------|------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming  | 5       | 95              |
| C23       | Integer Programming                      | 5       | 92              |
| C24       | Stochastic Processes                     | 4       | 86              |
| C25       | Simulation Modeling                      | 4       | 82              |
| C26       | Network Flows                            | 4       | 85              |
| C27       | Queueing Theory                          | 4       | 80              |
| C28       | Revenue Management                       | 4       | 88              |

So,
- $c_{\text{C22}} = 5$, $p_{\text{C22}} = 95$
- $c_{\text{C23}} = 5$, $p_{\text{C23}} = 92$
- $c_{\text{C24}} = 4$, $p_{\text{C24}} = 86$
- $c_{\text{C25}} = 4$, $p_{\text{C25}} = 82$
- $c_{\text{C26}} = 4$, $p_{\text{C26}} = 85$
- $c_{\text{C27}} = 4$, $p_{\text{C27}} = 80$
- $c_{\text{C28}} = 4$, $p_{\text{C28}} = 88$

##### Objective Function

\[
\max \sum_{i \in I} p_i x_i
\]

##### Constraints

1. Credit limit:
   \[
   \sum_{i \in I} c_i x_i \leq 20
   \]
2. Binary selection:
   \[
   x_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 95x_{\text{C22}} + 92x_{\text{C23}} + 86x_{\text{C24}} + 82x_{\text{C25}} + 85x_{\text{C26}} + 80x_{\text{C27}} + 88x_{\text{C28}} \\
\text{s.t.}\quad & 5x_{\text{C22}} + 5x_{\text{C23}} + 4x_{\text{C24}} + 4x_{\text{C25}} + 4x_{\text{C26}} + 4x_{\text{C27}} + 4x_{\text{C28}} \leq 20 \\
& x_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

Where $I = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$, with parameters as listed above.
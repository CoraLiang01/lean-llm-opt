##### Parameters

Let $N = \{\text{Depot}, A, B, C\}$.

The distance matrix $d_{ij}$ (in km) for $i, j \in N$ is:

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

##### Decision Variables

- $x_{ij} \in \{0,1\}$: 1 if the van travels directly from location $i$ to location $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in [1, 3]$: auxiliary variables for subtour elimination, for $i \in \{A, B, C\}$.

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Departure and Arrival Constraints:**
   - The van leaves each location exactly once:
     \[
     \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1, \quad \forall i \in N
     \]
   - The van enters each location exactly once:
     \[
     \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1, \quad \forall j \in N
     \]

2. **Subtour Elimination (MTZ) Constraints:**
   - For all $i, j \in \{A, B, C\}$, $i \neq j$:
     \[
     u_i - u_j + 3 x_{ij} \leq 2
     \]
   - $u_i \in [1, 3]$, for $i \in \{A, B, C\}$.

3. **Variable Domains:**
   - $x_{ij} \in \{0,1\}$, for all $i, j \in N$, $i \neq j$.

##### Parameters (explicit values):

\[
\begin{align*}
&d_{\text{Depot},A} = 28,\quad d_{\text{Depot},B} = 41,\quad d_{\text{Depot},C} = 63 \\
&d_{A,\text{Depot}} = 28,\quad d_{A,B} = 27,\quad d_{A,C} = 87 \\
&d_{B,\text{Depot}} = 41,\quad d_{B,A} = 27,\quad d_{B,C} = 81 \\
&d_{C,\text{Depot}} = 63,\quad d_{C,A} = 87,\quad d_{C,B} = 81 \\
\end{align*}
\]

##### Sequence Determination

To determine the optimal sequence, solve the above model. The optimal tour is the sequence of locations (starting and ending at Depot) that minimizes the total travel distance.
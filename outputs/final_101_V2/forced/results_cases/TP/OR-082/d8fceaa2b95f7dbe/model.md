##### Sets

Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (depot and customer locations).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$ (in km), for all $i, j \in N$:

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C \\
\hline
\text{Depot} & 0 & 28 & 41 & 63 \\
A & 28 & 0 & 27 & 87 \\
B & 41 & 27 & 0 & 81 \\
C & 63 & 87 & 81 & 0 \\
\end{array}
\]

##### Decision Variables

\[
x_{ij} = 
\begin{cases}
1 & \text{if the van travels directly from node } i \text{ to node } j \\
0 & \text{otherwise}
\end{cases}
\qquad \forall i, j \in N,\, i \neq j
\]

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Each node is departed exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \qquad \forall i \in N
   \]

2. **Each node is arrived at exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \qquad \forall j \in N
   \]

3. **Subtour elimination (MTZ formulation):**

   Introduce continuous variables $u_i$ for $i \in N$, with $u_{\text{Depot}} = 0$ and $1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$.

   \[
   u_i - u_j + 3 x_{ij} \leq 2 \qquad \forall i, j \in \{A, B, C\},\, i \neq j
   \]

4. **Binary variables:**
   \[
   x_{ij} \in \{0,1\} \qquad \forall i, j \in N,\, i \neq j
   \]

##### Data Used

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C \\
\hline
\text{Depot} & 0 & 28 & 41 & 63 \\
A & 28 & 0 & 27 & 87 \\
B & 41 & 27 & 0 & 81 \\
C & 63 & 87 & 81 & 0 \\
\end{array}
\]

##### Sequence of Visits

Enumerate all possible tours starting and ending at Depot, visiting A, B, and C exactly once:

- Depot $\to$ A $\to$ B $\to$ C $\to$ Depot: $28 + 27 + 81 + 63 = 199$
- Depot $\to$ A $\to$ C $\to$ B $\to$ Depot: $28 + 87 + 81 + 41 = 237$
- Depot $\to$ B $\to$ A $\to$ C $\to$ Depot: $41 + 27 + 87 + 63 = 218$
- Depot $\to$ B $\to$ C $\to$ A $\to$ Depot: $41 + 81 + 87 + 28 = 237$
- Depot $\to$ C $\to$ A $\to$ B $\to$ Depot: $63 + 87 + 27 + 41 = 218$
- Depot $\to$ C $\to$ B $\to$ A $\to$ Depot: $63 + 81 + 27 + 28 = 199$

The minimum total distance is $199$ km, and the optimal sequences are:

- Depot $\to$ A $\to$ B $\to$ C $\to$ Depot
- Depot $\to$ C $\to$ B $\to$ A $\to$ Depot

##### Complete Model

Minimize
\[
28x_{\text{Depot},A} + 41x_{\text{Depot},B} + 63x_{\text{Depot},C} + 28x_{A,\text{Depot}} + 27x_{A,B} + 87x_{A,C} + 41x_{B,\text{Depot}} + 27x_{B,A} + 81x_{B,C} + 63x_{C,\text{Depot}} + 87x_{C,A} + 81x_{C,B}
\]

Subject to:
\[
\begin{align*}
x_{\text{Depot},A} + x_{\text{Depot},B} + x_{\text{Depot},C} &= 1 \\
x_{A,\text{Depot}} + x_{A,B} + x_{A,C} &= 1 \\
x_{B,\text{Depot}} + x_{B,A} + x_{B,C} &= 1 \\
x_{C,\text{Depot}} + x_{C,A} + x_{C,B} &= 1 \\
x_{A,\text{Depot}} + x_{B,\text{Depot}} + x_{C,\text{Depot}} &= 1 \\
x_{\text{Depot},A} + x_{B,A} + x_{C,A} &= 1 \\
x_{\text{Depot},B} + x_{A,B} + x_{C,B} &= 1 \\
x_{\text{Depot},C} + x_{A,C} + x_{B,C} &= 1 \\
u_A - u_B + 3x_{A,B} &\leq 2 \\
u_B - u_A + 3x_{B,A} &\leq 2 \\
u_A - u_C + 3x_{A,C} &\leq 2 \\
u_C - u_A + 3x_{C,A} &\leq 2 \\
u_B - u_C + 3x_{B,C} &\leq 2 \\
u_C - u_B + 3x_{C,B} &\leq 2 \\
u_{\text{Depot}} = 0,\quad 1 \leq u_A, u_B, u_C \leq 3 \\
x_{ij} \in \{0,1\} \quad \forall i \neq j \in N
\end{align*}
\]

Optimal tour: Depot $\to$ A $\to$ B $\to$ C $\to$ Depot or Depot $\to$ C $\to$ B $\to$ A $\to$ Depot, with total distance $199$ km.
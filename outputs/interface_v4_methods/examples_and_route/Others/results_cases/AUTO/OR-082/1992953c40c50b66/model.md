##### Sets

Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (Depot and three customers).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$, for all $i, j \in N$:

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

$x_{ij} \in \{0,1\}$: 1 if the van travels directly from node $i$ to node $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.

$u_i \in \mathbb{Z}$: Subtour elimination variable for $i \in \{A, B, C\}$.

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Leave each node exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]

2. **Enter each node exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]

3. **Subtour elimination (MTZ):**
   \[
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\ i \neq j
   \]
   \[
   1 \leq u_i \leq 3 \quad \forall i \in \{A, B, C\}
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\ i \neq j
   \]
   \[
   u_i \in \mathbb{Z} \quad \forall i \in \{A, B, C\}
   \]

##### Data Used

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ as given above (all pairwise distances between Depot, A, B, C, from DistanceMatrix.csv, source order preserved).

##### Sequence of Visits

To determine the optimal sequence, enumerate all possible tours starting and ending at Depot and visiting A, B, C exactly once:

- Depot → A → B → C → Depot: $28 + 27 + 81 + 63 = 199$
- Depot → A → C → B → Depot: $28 + 87 + 81 + 41 = 237$
- Depot → B → A → C → Depot: $41 + 27 + 87 + 63 = 218$
- Depot → B → C → A → Depot: $41 + 81 + 87 + 28 = 237$
- Depot → C → A → B → Depot: $63 + 87 + 27 + 41 = 218$
- Depot → C → B → A → Depot: $63 + 81 + 27 + 28 = 199$

The minimum total distance is $199$ km, and there are two optimal sequences:

- Depot → A → B → C → Depot
- Depot → C → B → A → Depot

##### Complete Model

\[
\begin{align*}
\min\ & 28x_{\text{Depot},A} + 41x_{\text{Depot},B} + 63x_{\text{Depot},C} + 28x_{A,\text{Depot}} + 27x_{A,B} + 87x_{A,C} \\
& + 41x_{B,\text{Depot}} + 27x_{B,A} + 81x_{B,C} + 63x_{C,\text{Depot}} + 87x_{C,A} + 81x_{C,B} \\
\text{s.t. } & x_{\text{Depot},A} + x_{\text{Depot},B} + x_{\text{Depot},C} = 1 \\
& x_{A,\text{Depot}} + x_{A,B} + x_{A,C} = 1 \\
& x_{B,\text{Depot}} + x_{B,A} + x_{B,C} = 1 \\
& x_{C,\text{Depot}} + x_{C,A} + x_{C,B} = 1 \\
& x_{\text{Depot},A} + x_{B,A} + x_{C,A} = 1 \\
& x_{A,\text{Depot}} + x_{B,\text{Depot}} + x_{C,\text{Depot}} = 1 \\
& x_{\text{Depot},B} + x_{A,B} + x_{C,B} = 1 \\
& x_{\text{Depot},C} + x_{A,C} + x_{B,C} = 1 \\
& u_A - u_B + 3x_{A,B} \leq 2 \\
& u_A - u_C + 3x_{A,C} \leq 2 \\
& u_B - u_A + 3x_{B,A} \leq 2 \\
& u_B - u_C + 3x_{B,C} \leq 2 \\
& u_C - u_A + 3x_{C,A} \leq 2 \\
& u_C - u_B + 3x_{C,B} \leq 2 \\
& 1 \leq u_A \leq 3 \\
& 1 \leq u_B \leq 3 \\
& 1 \leq u_C \leq 3 \\
& x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j \\
& u_i \in \mathbb{Z} \quad \forall i \in \{A, B, C\}
\end{align*}
\]

Optimal tour(s): Depot → A → B → C → Depot or Depot → C → B → A → Depot, with total distance 199 km.
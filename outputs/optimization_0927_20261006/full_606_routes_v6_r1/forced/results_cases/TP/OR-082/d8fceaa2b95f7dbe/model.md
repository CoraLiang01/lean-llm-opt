##### Sets
Let $N = \{\text{Depot}, A, B, C\}$.

##### Parameters
Let $d_{ij}$ be the distance from location $i$ to location $j$ (in km), for all $i, j \in N$:

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
$x_{ij} \in \{0,1\}$: 1 if the van travels directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.

$u_i \in \mathbb{Z}$: auxiliary variables for subtour elimination, for all $i \in N$, $i \neq \text{Depot}$.

##### Objective
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Departure from each node:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]

2. **Arrival at each node:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]

3. **Subtour elimination (MTZ):**
   \[
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in N,\, i \neq j,\, i \neq \text{Depot},\, j \neq \text{Depot}
   \]
   \[
   1 \leq u_i \leq 3 \quad \forall i \in N,\, i \neq \text{Depot}
   \]

4. **Binary variables:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   \]

##### Data

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

Let the optimal solution be the sequence of nodes starting and ending at Depot, visiting each of $A$, $B$, $C$ exactly once, and minimizing the total travel distance as per the above model.
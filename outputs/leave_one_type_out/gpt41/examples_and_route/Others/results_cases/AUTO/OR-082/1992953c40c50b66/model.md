##### Sets

Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (depot and customer locations).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$ for all $i, j \in N$, given by:

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

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Departure from each node exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]

2. **Arrival at each node exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]

3. **Subtour elimination (MTZ formulation):**

   Introduce continuous variables $u_i$ for $i \in N$, with $u_{\text{Depot}} = 0$ and $1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$.

   \[
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\ i \neq j
   \]

4. **Binary variables:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\ i \neq j
   \]

##### Data Used

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ as in the table above, extracted from DistanceMatrix.csv.

##### Sequence of Visits

Let the optimal solution be the sequence of nodes starting and ending at Depot, visiting each of $A$, $B$, $C$ exactly once, that minimizes the total travel distance as per the model above.
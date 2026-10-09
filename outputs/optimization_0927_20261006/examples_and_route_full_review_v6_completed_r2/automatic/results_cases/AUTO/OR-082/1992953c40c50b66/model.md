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

$u_i \in [1, |N|-1]$: auxiliary variables for subtour elimination, for all $i \in N$, $i \neq \text{Depot}$.

##### Objective Function
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

3. **Subtour elimination (MTZ constraints):**
   \[
   u_i - u_j + (|N|-1) x_{ij} \leq |N|-2 \quad \forall i, j \in N,\, i \neq j,\, i \neq \text{Depot},\, j \neq \text{Depot}
   \]
   where $u_i \in [1, |N|-1]$ for $i \in N,\, i \neq \text{Depot}$.

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in [1,3] \quad \forall i \in \{A, B, C\}
   \]

##### Data Used

- $N = \{\text{Depot}, A, B, C\}$
- Distance matrix (all $d_{ij}$ as above, from DistanceMatrix.csv)

##### Model Summary

Minimize total travel distance by selecting a tour that starts and ends at the Depot, visits each of A, B, and C exactly once, and does not contain subtours. The optimal sequence is the order of visits corresponding to the $x_{ij}$ variables set to 1 in the optimal solution.
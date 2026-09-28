##### Sets

Let $N = \{\text{Depot}, A, B, C\}$ be the set of locations (including the depot and three customers).

##### Parameters

Let $d_{ij}$ be the distance (in kilometres) from location $i$ to location $j$, for all $i, j \in N$.

From the data:

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
1 & \text{if the van travels directly from location } i \text{ to location } j \\
0 & \text{otherwise}
\end{cases}
\qquad \forall i, j \in N,\, i \neq j
\]

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Departure from each location exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \qquad \forall i \in N
   \]

2. **Arrival at each location exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \qquad \forall j \in N
   \]

3. **Subtour elimination (Miller-Tucker-Zemlin for small $N$):**

   Introduce continuous variables $u_i$ for $i \in N$, with $u_{\text{Depot}} = 0$ and $1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$.

   \[
   u_i - u_j + 3 x_{ij} \leq 2 \qquad \forall i, j \in \{A, B, C\},\, i \neq j
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \qquad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in \mathbb{R} \qquad \forall i \in N
   \]

##### Parameters (explicit values)

\[
\begin{align*}
&N = \{\text{Depot}, A, B, C\} \\
&d_{ij} = \text{as in the table above}
\end{align*}
\]

##### Model Summary

- The van starts at Depot, visits each of $A$, $B$, $C$ exactly once, and returns to Depot.
- The objective is to minimize the total travel distance.
- The variables $x_{ij}$ indicate the sequence of visits.
- Subtour elimination constraints prevent disconnected cycles.

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

##### Note

To determine the optimal sequence, solve the above model. For this small instance, all possible tours can be enumerated:

- Depot $\to$ A $\to$ B $\to$ C $\to$ Depot: $28 + 27 + 81 + 63 = 199$
- Depot $\to$ A $\to$ C $\to$ B $\to$ Depot: $28 + 87 + 81 + 41 = 237$
- Depot $\to$ B $\to$ A $\to$ C $\to$ Depot: $41 + 27 + 87 + 63 = 218$
- Depot $\to$ B $\to$ C $\to$ A $\to$ Depot: $41 + 81 + 87 + 28 = 237$
- Depot $\to$ C $\to$ A $\to$ B $\to$ Depot: $63 + 87 + 27 + 41 = 218$
- Depot $\to$ C $\to$ B $\to$ A $\to$ Depot: $63 + 81 + 27 + 28 = 199$

Thus, the minimum total distance is $199$ km, and the optimal sequences are:

- Depot $\to$ A $\to$ B $\to$ C $\to$ Depot
- Depot $\to$ C $\to$ B $\to$ A $\to$ Depot

Either sequence is optimal.
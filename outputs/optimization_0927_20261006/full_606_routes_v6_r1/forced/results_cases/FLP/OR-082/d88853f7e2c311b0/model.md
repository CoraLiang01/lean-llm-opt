##### Sets

Let $N = \{\text{Depot}, A, B, C\}$ be the set of all nodes (depot and customers).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$, for all $i, j \in N$.

From the retrieved DistanceMatrix.csv, the relevant submatrix is:

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

So,
- $d_{\text{Depot},A} = 28$, $d_{\text{Depot},B} = 41$, $d_{\text{Depot},C} = 63$
- $d_{A,\text{Depot}} = 28$, $d_{A,B} = 27$, $d_{A,C} = 87$
- $d_{B,\text{Depot}} = 41$, $d_{B,A} = 27$, $d_{B,C} = 81$
- $d_{C,\text{Depot}} = 63$, $d_{C,A} = 87$, $d_{C,B} = 81$

##### Decision Variables

For all $i, j \in N$, $i \neq j$:
- $x_{ij} \in \{0,1\}$: 1 if the van travels directly from node $i$ to node $j$, 0 otherwise.

For all $i \in N$, $i \neq \text{Depot}$:
- $u_i \in \{2,3,4\}$: the position of node $i$ in the tour (Miller-Tucker-Zemlin subtour elimination).

##### Objective Function

\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

##### Constraints

1. **Each node is departed exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1, \quad \forall i \in N
   \]

2. **Each node is arrived at exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1, \quad \forall j \in N
   \]

3. **Subtour elimination (MTZ):** For all $i, j \in N$, $i \neq j$, $i \neq \text{Depot}$, $j \neq \text{Depot}$:
   \[
   u_i - u_j + 3 x_{ij} \leq 2
   \]
   where $u_i \in \{2,3,4\}$ for $i \in \{A,B,C\}$.

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in \{2,3,4\}, \quad \forall i \in \{A,B,C\}
   \]

##### Parameters (explicit values)

\[
\begin{align*}
&d_{\text{Depot},A} = 28,\quad d_{\text{Depot},B} = 41,\quad d_{\text{Depot},C} = 63 \\
&d_{A,\text{Depot}} = 28,\quad d_{A,B} = 27,\quad d_{A,C} = 87 \\
&d_{B,\text{Depot}} = 41,\quad d_{B,A} = 27,\quad d_{B,C} = 81 \\
&d_{C,\text{Depot}} = 63,\quad d_{C,A} = 87,\quad d_{C,B} = 81 \\
\end{align*}
\]

##### Model Summary

Minimize total travel distance by selecting a tour that starts and ends at the depot, visits each of $A$, $B$, and $C$ exactly once, and does not form subtours.

##### Retrieved Information

{
  "nodes": ["Depot", "A", "B", "C"],
  "distance_matrix": {
    "Depot": {"Depot": 0, "A": 28, "B": 41, "C": 63},
    "A": {"Depot": 28, "A": 0, "B": 27, "C": 87},
    "B": {"Depot": 41, "A": 27, "B": 0, "C": 81},
    "C": {"Depot": 63, "A": 87, "B": 81, "C": 0}
  }
}
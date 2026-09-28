Let $N = \{\text{Depot}, A, B, C\}$ be the set of locations (Depot and three customers). Let $d_{ij}$ denote the distance from node $i$ to node $j$, as given in the table below. Define binary variables $x_{ij}$, where $x_{ij} = 1$ if the route travels directly from $i$ to $j$, and $0$ otherwise.

Minimize total travel distance:
$$
\min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}
$$

Subject to:

1. Each location (including the depot) is departed from exactly once:
$$
\sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N
$$

2. Each location (including the depot) is arrived at exactly once:
$$
\sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N
$$

3. Subtour elimination (Miller-Tucker-Zemlin constraints): Introduce continuous variables $u_i$ for $i \in \{A, B, C\}$, with $1 \leq u_i \leq 3$.
$$
u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\, i \neq j
$$

4. Binary variables:
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
$$

5. $u_{\text{Depot}}$ is fixed (e.g., $u_{\text{Depot}} = 0$).

Distance matrix (retrieved data, restricted to relevant nodes):

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

Where $d_{ij}$ is the value in row $i$, column $j$.

Decision: The optimal sequence of visits is the tour corresponding to the solution of this model that minimizes total travel distance.
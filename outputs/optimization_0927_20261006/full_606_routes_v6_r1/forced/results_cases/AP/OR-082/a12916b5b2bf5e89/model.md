##### Objective Function:

$\quad \min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}$

where:
- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ is the distance from node $i$ to node $j$ (see data below)
- $x_{ij} = 1$ if the route goes directly from $i$ to $j$, $0$ otherwise

##### Constraints:

1. **Visit Each Node Exactly Once (except depot, which is start and end):**

$\sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N$

$\sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N$

2. **Subtour Elimination (MTZ Formulation):**

Let $u_i$ be an auxiliary variable for $i \in \{A, B, C\}$ (not defined for Depot), with $1 \leq u_i \leq 3$.

$u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\, i \neq j$

3. **Variable Domains:**

$x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j$

$u_i \in \mathbb{Z},\ 1 \leq u_i \leq 3 \quad \forall i \in \{A, B, C\}$

##### Retrieved Information

{
  "nodes": ["Depot", "A", "B", "C"],
  "distance": {
    "Depot": {"Depot": 0, "A": 28, "B": 41, "C": 63},
    "A": {"Depot": 28, "A": 0, "B": 27, "C": 87},
    "B": {"Depot": 41, "A": 27, "B": 0, "C": 81},
    "C": {"Depot": 63, "A": 87, "B": 81, "C": 0}
  }
}

##### Decision Variables

- $x_{ij}$: binary, 1 if the van travels directly from node $i$ to node $j$, 0 otherwise, for all $i, j \in \{\text{Depot}, A, B, C\},\, i \neq j$
- $u_i$: integer, position in the tour for $i \in \{A, B, C\}$

##### Full Distance Matrix (for reference):

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

##### Model Summary

- Minimize total travel distance for a tour starting and ending at Depot, visiting A, B, and C exactly once each.
- Each location is visited once, and the van returns to the depot.
- Subtours are eliminated using the MTZ constraints.
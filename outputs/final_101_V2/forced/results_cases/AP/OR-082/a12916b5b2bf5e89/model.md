##### Objective Function:

$\quad \min \sum_{i \in N} \sum_{j \in N, j \neq i} d_{ij} x_{ij}$

where:
- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ is the distance from node $i$ to node $j$ (see data below)
- $x_{ij} = 1$ if the route goes directly from $i$ to $j$, $0$ otherwise

##### Constraints:

1. **Visit Each Node Exactly Once (except Depot):**

$\sum_{j \in N, j \neq i} x_{ij} = 1 \quad \forall i \in N, i \neq \text{Depot}$

$\sum_{i \in N, i \neq j} x_{ij} = 1 \quad \forall j \in N, j \neq \text{Depot}$

2. **Depot Departure and Return:**

$\sum_{j \in N, j \neq \text{Depot}} x_{\text{Depot},j} = 1$

$\sum_{i \in N, i \neq \text{Depot}} x_{i,\text{Depot}} = 1$

3. **Subtour Elimination:**

For all $i, j \in \{A, B, C\}, i \neq j$:

$u_i - u_j + 3 x_{ij} \leq 2$

where $u_i$ are integer variables with $1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$

4. **Variable Domains:**

$x_{ij} \in \{0,1\} \quad \forall i, j \in N, i \neq j$

$u_i \in \mathbb{Z}, 1 \leq u_i \leq 3 \quad \forall i \in \{A, B, C\}$

##### Retrieved Information

{
  "distance": {
    "Depot": {"Depot": 0, "A": 28, "B": 41, "C": 63},
    "A": {"Depot": 28, "A": 0, "B": 27, "C": 87},
    "B": {"Depot": 41, "A": 27, "B": 0, "C": 81},
    "C": {"Depot": 63, "A": 87, "B": 81, "C": 0}
  },
  "nodes": ["Depot", "A", "B", "C"]
}

##### Enumerated Solution

All possible sequences and their total distances:

| Sequence                      | Total Distance (km) |
|-------------------------------|---------------------|
| Depot → A → B → C → Depot     | 199                 |
| Depot → A → C → B → Depot     | 237                 |
| Depot → B → A → C → Depot     | 218                 |
| Depot → B → C → A → Depot     | 237                 |
| Depot → C → A → B → Depot     | 218                 |
| Depot → C → B → A → Depot     | 199                 |

**Minimum total distance:** 199 km

**Optimal sequences:** Depot → A → B → C → Depot or Depot → C → B → A → Depot

##### Model Parameters

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ as given above

##### Decision Variables

- $x_{ij} \in \{0,1\}$ for all $i, j \in N, i \neq j$
- $u_i \in \mathbb{Z}, 1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$

##### Objective

Minimise total travel distance for a tour starting and ending at the Depot, visiting each of A, B, and C exactly once.
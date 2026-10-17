##### Objective Function:

$\quad \min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}$

##### Sets and Parameters:

- $N = \{\text{Depot},\, A,\, B,\, C\}$
- $d_{ij}$: Distance from node $i$ to node $j$, as given below.

##### Decision Variables:

- $x_{ij} \in \{0,1\}$: 1 if the route goes directly from node $i$ to node $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in \mathbb{Z}$: Subtour elimination variable for $i \in N$, $i \neq \text{Depot}$.

##### Constraints:

1. **Departure Constraints:**
   $$
   \sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N
   $$
   (Exactly one outgoing arc from each node.)

2. **Arrival Constraints:**
   $$
   \sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N
   $$
   (Exactly one incoming arc to each node.)

3. **Subtour Elimination Constraints (MTZ):**
   $$
   u_i - u_j + 3\, x_{ij} \leq 2 \quad \forall i, j \in N,\, i \neq j,\, i \neq \text{Depot},\, j \neq \text{Depot}
   $$
   $$
   1 \leq u_i \leq 3 \quad \forall i \in N,\, i \neq \text{Depot}
   $$
   (Eliminates subtours for 4-node TSP.)

4. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   $$
   $$
   u_i \in \mathbb{Z} \quad \forall i \in N,\, i \neq \text{Depot}
   $$

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

##### Explicit Distance Matrix Used

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

##### Notes

- The model ensures the van starts and ends at the depot, visits each customer location (A, B, C) exactly once, and minimizes the total travel distance.
- The subtour elimination constraints (MTZ) prevent disconnected cycles.
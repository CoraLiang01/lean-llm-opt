##### Objective Function:

$\quad \min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}$

##### Constraints

###### 1. Departure and Arrival Constraints (each node is entered and exited exactly once):

$\sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N$

$\sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N$

###### 2. Subtour Elimination Constraints (MTZ formulation):

For all $i, j \in N \setminus \{\text{Depot}\}$, $i \neq j$:

$u_i - u_j + 3 x_{ij} \leq 2$

$u_{\text{Depot}} = 0$

$1 \leq u_i \leq 3 \quad \forall i \in N \setminus \{\text{Depot}\}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j$

$u_i \in \mathbb{Z} \quad \forall i \in N \setminus \{\text{Depot}\}$

---

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

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ is the distance from node $i$ to node $j$, as given above.
- $x_{ij} = 1$ if the route goes directly from node $i$ to node $j$, $0$ otherwise.
- $u_i$ are auxiliary variables for subtour elimination (for $i \in \{A, B, C\}$).

---

##### Full Distance Matrix Used

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |
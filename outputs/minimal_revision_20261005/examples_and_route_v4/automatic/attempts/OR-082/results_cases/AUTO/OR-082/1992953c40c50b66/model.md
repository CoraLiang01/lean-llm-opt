##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all $i,j \in N$, where $N = \{\text{Depot}, A, B, C\}$.

- $x_{ij} = 1$ if the van travels directly from node $i$ to node $j$, $0$ otherwise.

Let $u_i \in \mathbb{Z}$ for all $i \in \{A, B, C\}$ (subtour elimination variables).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$, as given in DistanceMatrix.csv (table_id: file_0_view_0, columns: Unnamed: 0, Depot, A, B, C).

##### Objective Function

Minimize the total travel distance:
$$
\min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}
$$

##### Constraints

1. **Departure from each node:**
   $$
   \sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N
   $$
2. **Arrival to each node:**
   $$
   \sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N
   $$
3. **Subtour elimination (Miller-Tucker-Zemlin for $n=4$):**
   $$
   u_i - u_j + 3\, x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\, i \neq j
   $$
   $$
   1 \leq u_i \leq 3 \quad \forall i \in \{A, B, C\}
   $$
4. **No self-loops:**
   $$
   x_{ii} = 0 \quad \forall i \in N
   $$

##### Data Mapping

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$: Distance from node $i$ (row, column Unnamed: 0) to node $j$ (column), from table_id: file_0_view_0, columns: Unnamed: 0, Depot, A, B, C, using only these four nodes.
- All $x_{ij}$ and $u_i$ as defined above.

##### Notes

- The tour starts and ends at Depot, visiting each of $A$, $B$, $C$ exactly once.
- All distances $d_{ij}$ are taken directly from the corresponding entries in DistanceMatrix.csv (file_0_view_0).
- The model is a standard asymmetric TSP for 4 nodes.
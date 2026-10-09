## Mathematical Model

### Sets
- $N = \{1,2,\ldots,15\}$: set of locations (with 1 as the depot/start/end).

### Parameters
- $d_{ij}$: distance from location $i$ to $j$, for all $i, j \in N$, $i \neq j$.
  - Data Mapping: $d_{ij}$ is given by the symmetric matrix in table_id: file_0_view_0, with rows and columns corresponding to locations $i$ and $j$ (indices $1$ to $15$).

### Decision Variables
- $x_{ij} \in \{0,1\}$: 1 if the route goes directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in [2,15]$: position of node $i$ in the tour (for subtour elimination), for all $i \in N$, $i \neq 1$.

### Objective
Minimize total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

### Constraints

1. **Leave each node exactly once:**
   $$
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   $$

2. **Enter each node exactly once:**
   $$
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   $$

3. **Subtour elimination (MTZ constraints):**
   $$
   u_i - u_j + 15\, x_{ij} \leq 14 \quad \forall i, j \in N,\, i \neq j,\, i \neq 1,\, j \neq 1
   $$
   $$
   2 \leq u_i \leq 15 \quad \forall i \in N,\, i \neq 1
   $$

4. **Variable domains:**
   $$
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   $$
   $$
   u_i \in [2,15] \quad \forall i \in N,\, i \neq 1
   $$

### Data Mapping

- $d_{ij}$: For $i < j$, $d_{ij}$ is the value in row $i-1$, column "$j$" of table_id: file_0_view_0. For $i > j$, $d_{ij} = d_{ji}$ (symmetry).
- $x_{ij}$: Binary variable for each ordered pair $(i,j)$, $i \neq j$.
- $u_i$: Continuous variable for each $i \in \{2,\ldots,15\}$.

### Notes

- The tour starts and ends at location 1.
- All locations are visited exactly once.
- The MTZ constraints prevent subtours.

This model is a standard symmetric TSP integer program using the provided distance matrix.
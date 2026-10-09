##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all $i, j \in N$, where $N = \{\text{Depot}, A, B, C\}$.
- $x_{ij} = 1$ if the van travels directly from node $i$ to node $j$, $0$ otherwise.

Let $u_i \in \mathbb{Z}$ for all $i \in \{A, B, C\}$ (subtour elimination variables).

##### Parameters

Let $d_{ij}$ be the distance from node $i$ to node $j$, as given in table_id file_0_view_0, columns and rows indexed by $N = \{\text{Depot}, A, B, C\}$.

##### Objective Function

Minimize the total travel distance:
$$
\min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}
$$

##### Constraints

1. **Departure from each node:** For all $i \in N$, $\sum_{j \in N,\, j \neq i} x_{ij} = 1$
2. **Arrival to each node:** For all $j \in N$, $\sum_{i \in N,\, i \neq j} x_{ij} = 1$
3. **Subtour elimination (MTZ):** For all $i, j \in \{A, B, C\},\, i \neq j$,
   $$
   u_i - u_j + 3 x_{ij} \leq 2
   $$
   with $u_i \in \{1,2,3\}$ for $i \in \{A, B, C\}$.
4. **No self-loops:** $x_{ii} = 0$ for all $i \in N$.

##### Data Mapping

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$: distance from node $i$ (row) to node $j$ (column) in table_id file_0_view_0, using only the rows and columns for Depot, A, B, C.
- All variables and constraints are indexed over these four nodes.

##### Notes

- The model is a standard asymmetric TSP for four nodes (Depot, A, B, C).
- The optimal sequence is the tour minimizing total distance, starting and ending at Depot, visiting A, B, and C exactly once each.
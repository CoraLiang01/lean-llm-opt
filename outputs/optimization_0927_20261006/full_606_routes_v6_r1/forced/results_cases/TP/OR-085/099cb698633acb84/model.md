##### Sets
Let $N = \{1,2,\ldots,15\}$ be the set of locations (with location 1 as the depot/start/end).

##### Parameters
Let $d_{ij}$ be the distance from location $i$ to location $j$, for all $i,j \in N$, $i \neq j$.

The distance matrix $[d_{ij}]$ is:

|   | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| 1 |   | 67 | 55 | 80 | 21 | 77 | 78 | 74 | 85 | 28 | 55 | 53 | 66 | 89 | 78 |
| 2 | 67 |   | 38 | 29 | 68 | 36 | 62 | 54 | 49 | 92 | 37 | 51 | 38 | 82 | 31 |
| 3 | 55 | 38 |   | 28 | 44 | 27 | 56 | 34 | 33 | 68 | 70 | 55 | 46 | 32 | 40 |
| 4 | 80 | 29 | 28 |   | 21 | 51 | 46 | 48 | 31 | 55 | 68 | 85 | 58 | 56 | 22 |
| 5 | 21 | 68 | 44 | 21 |   | 42 | 57 | 31 | 55 | 79 | 49 | 70 | 43 | 55 | 78 |
| 6 | 77 | 36 | 27 | 51 | 42 |   | 63 | 41 | 39 | 52 | 76 | 54 | 59 | 44 | 76 |
| 7 | 78 | 62 | 56 | 46 | 57 | 63 |   | 38 | 35 | 37 | 55 | 54 | 51 | 14 | 64 |
| 8 | 74 | 54 | 34 | 48 | 31 | 41 | 38 |   | 53 | 24 | 60 | 42 | 31 | 42 | 27 |
| 9 | 85 | 49 | 33 | 31 | 55 | 39 | 35 | 53 |   | 88 | 28 | 65 | 12 | 63 | 45 |
| 10 | 28 | 92 | 68 | 55 | 79 | 52 | 37 | 24 | 88 |   | 84 | 40 | 43 | 81 | 37 |
| 11 | 55 | 37 | 70 | 68 | 49 | 76 | 55 | 60 | 28 | 84 |   | 41 | 38 | 56 | 35 |
| 12 | 53 | 51 | 55 | 85 | 70 | 54 | 54 | 42 | 65 | 40 | 41 |   | 65 | 47 | 38 |
| 13 | 66 | 38 | 46 | 58 | 43 | 59 | 51 | 31 | 12 | 43 | 38 | 65 |   | 35 | 77 |
| 14 | 89 | 82 | 32 | 56 | 55 | 44 | 14 | 42 | 63 | 81 | 56 | 47 | 35 |   | 54 |
| 15 | 78 | 31 | 40 | 22 | 78 | 76 | 64 | 27 | 45 | 37 | 35 | 38 | 77 | 54 |   |

(Empty entries correspond to $i=j$ and are not used.)

##### Decision Variables

$x_{ij} \in \{0,1\}$: 1 if the tour goes directly from location $i$ to location $j$, 0 otherwise, for all $i,j \in N$, $i \neq j$.

$u_i \in [2,15]$: the position of location $i$ in the tour (for $i=2,\ldots,15$), used for subtour elimination.

##### Objective

Minimize the total distance traveled:
$$
\min \sum_{i=1}^{15} \sum_{\substack{j=1 \\ j \neq i}}^{15} d_{ij} x_{ij}
$$

##### Constraints

1. Each location is departed exactly once:
$$
\sum_{\substack{j=1 \\ j \neq i}}^{15} x_{ij} = 1 \quad \forall i \in N
$$

2. Each location is arrived at exactly once:
$$
\sum_{\substack{i=1 \\ i \neq j}}^{15} x_{ij} = 1 \quad \forall j \in N
$$

3. Subtour elimination (Miller-Tucker-Zemlin, for $i,j \in \{2,\ldots,15\},\ i \neq j$):
$$
u_i - u_j + 14\, x_{ij} \leq 13 \quad \forall i,j \in \{2,\ldots,15\},\ i \neq j
$$

4. Variable domains:
$$
x_{ij} \in \{0,1\} \quad \forall i,j \in N,\ i \neq j
$$
$$
u_i \in [2,15] \quad \forall i \in \{2,\ldots,15\}
$$

##### Notes

- The tour starts and ends at location 1.
- The distance matrix is symmetric: $d_{ij} = d_{ji}$ for all $i \neq j$.
- The $u_i$ variables prevent subtours by enforcing a unique order for each location after the depot.

##### Data

The full $d_{ij}$ matrix is as given above, with all coefficients and identifiers preserved from the source.
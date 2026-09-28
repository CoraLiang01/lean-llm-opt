Let $N = 15$ be the number of locations, indexed by $i, j = 1, \ldots, 15$. Let $d_{ij}$ be the distance from location $i$ to location $j$, as given in the table below. Define binary variables $x_{ij}$, where $x_{ij} = 1$ if the route goes directly from $i$ to $j$, and $0$ otherwise. Let $u_i$ be continuous variables for subtour elimination (Miller-Tucker-Zemlin formulation), for $i = 2, \ldots, 15$.

Minimize total distance:
$$
\min \sum_{i=1}^{15} \sum_{\substack{j=1 \\ j \neq i}}^{15} d_{ij} x_{ij}
$$

Subject to:

1. Each location is departed exactly once:
$$
\sum_{\substack{j=1 \\ j \neq i}}^{15} x_{ij} = 1 \quad \forall i = 1, \ldots, 15
$$

2. Each location is arrived at exactly once:
$$
\sum_{\substack{i=1 \\ i \neq j}}^{15} x_{ij} = 1 \quad \forall j = 1, \ldots, 15
$$

3. Subtour elimination (MTZ constraints):
$$
u_i - u_j + 15 x_{ij} \leq 14 \quad \forall i, j = 2, \ldots, 15,\ i \neq j
$$
$$
1 \leq u_i \leq 14 \quad \forall i = 2, \ldots, 15
$$

4. Binary variables:
$$
x_{ij} \in \{0,1\} \quad \forall i, j = 1, \ldots, 15,\ i \neq j
$$

5. Start and end at location 1.

---

#### Distance Matrix $d_{ij}$ (from 20.csv, $i$ = row, $j$ = column):

|   | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| 1 | 0 | 67 | 55 | 80 | 21 | 77 | 78 | 74 | 85 | 28 | 55 | 53 | 66 | 89 | 78 |
| 2 | 67 | 0 | 38 | 29 | 68 | 36 | 62 | 54 | 49 | 92 | 37 | 51 | 38 | 82 | 31 |
| 3 | 55 | 38 | 0 | 28 | 44 | 27 | 56 | 34 | 33 | 68 | 70 | 55 | 46 | 32 | 40 |
| 4 | 80 | 29 | 28 | 0 | 21 | 51 | 46 | 48 | 31 | 55 | 68 | 85 | 58 | 56 | 22 |
| 5 | 21 | 68 | 44 | 21 | 0 | 42 | 57 | 31 | 55 | 79 | 49 | 70 | 43 | 55 | 78 |
| 6 | 77 | 36 | 27 | 51 | 42 | 0 | 63 | 41 | 39 | 52 | 76 | 54 | 59 | 44 | 76 |
| 7 | 78 | 62 | 56 | 46 | 57 | 63 | 0 | 38 | 35 | 37 | 55 | 54 | 51 | 14 | 64 |
| 8 | 74 | 54 | 34 | 48 | 31 | 41 | 38 | 0 | 53 | 24 | 60 | 42 | 31 | 42 | 27 |
| 9 | 85 | 49 | 33 | 31 | 55 | 39 | 35 | 53 | 0 | 88 | 28 | 65 | 12 | 63 | 45 |
| 10 | 28 | 92 | 68 | 55 | 79 | 52 | 37 | 24 | 88 | 0 | 84 | 40 | 43 | 81 | 37 |
| 11 | 55 | 37 | 70 | 68 | 49 | 76 | 55 | 60 | 28 | 84 | 0 | 41 | 38 | 56 | 35 |
| 12 | 53 | 51 | 55 | 85 | 70 | 54 | 54 | 42 | 65 | 40 | 41 | 0 | 65 | 47 | 38 |
| 13 | 66 | 38 | 46 | 58 | 43 | 59 | 51 | 31 | 12 | 43 | 38 | 65 | 0 | 35 | 77 |
| 14 | 89 | 82 | 32 | 56 | 55 | 44 | 14 | 42 | 63 | 81 | 56 | 47 | 35 | 0 | 54 |
| 15 | 78 | 31 | 40 | 22 | 78 | 76 | 64 | 27 | 45 | 37 | 35 | 38 | 77 | 54 | 0 |

Where $d_{ij}$ is symmetric and $d_{ii} = 0$ for all $i$.

---

**Variables:**
- $x_{ij} \in \{0,1\}$ for $i \neq j$, $i,j = 1,\ldots,15$
- $u_i \in [1,14]$ for $i = 2,\ldots,15$

**Objective:**
Minimize the total distance traveled, starting and ending at location 1, visiting each location exactly once.
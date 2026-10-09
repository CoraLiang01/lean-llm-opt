Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

#### Sets and Indices

- $i \in \{\text{SectionID }1, 2, 3, 4, 5, 6, 7, 8\}$
- $j \in \{\text{ProductName }1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

#### Parameters

- $p_j$: Value (price) of product $j$
- $w_j$: Weight (shelf space requirement) of product $j$
- $C_i$: Capacity of section $i$

Parameter values (from the data):

| SectionID | Capacity |
|-----------|----------|
| 1         | 100      |
| 2         | 150      |
| 3         | 120      |
| 4         | 130      |
| 5         | 90       |
| 6         | 110      |
| 7         | 160      |
| 8         | 140      |

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 10    | 2      |
| 2           | 15    | 3      |
| 3           | 8     | 1      |
| 4           | 12    | 2      |
| 5           | 20    | 4      |
| 6           | 25    | 5      |
| 7           | 5     | 1      |
| 8           | 30    | 6      |
| 9           | 18    | 3      |
| 10          | 22    | 4      |

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} p_j \cdot x_{ij}
\]

**Subject to:**

- Section capacity constraints (for each section $i$):
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
\]

- Integer and nonnegativity constraints:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\; j \in \{1,\ldots,10\}
\]

#### Parameter Tables

**Section Capacities:**

| SectionID | Capacity |
|-----------|----------|
| 1         | 100      |
| 2         | 150      |
| 3         | 120      |
| 4         | 130      |
| 5         | 90       |
| 6         | 110      |
| 7         | 160      |
| 8         | 140      |

**Product Values and Weights:**

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 10    | 2      |
| 2           | 15    | 3      |
| 3           | 8     | 1      |
| 4           | 12    | 2      |
| 5           | 20    | 4      |
| 6           | 25    | 5      |
| 7           | 5     | 1      |
| 8           | 30    | 6      |
| 9           | 18    | 3      |
| 10          | 22    | 4      |
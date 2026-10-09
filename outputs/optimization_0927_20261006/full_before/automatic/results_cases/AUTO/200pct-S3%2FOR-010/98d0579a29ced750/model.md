Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $i$ indexes SectionID from "capacity.csv" and $j$ indexes ProductName from "products.csv". All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of sections (SectionID from capacity.csv): $S = \{1,2,3,4,5,6,7,8\}$
- $P$ = set of products (ProductName from products.csv): $P = \{1,2,3,4,5,6,7,8,9,10\}$
- $c_i$ = Capacity of section $i$ (from capacity.csv, column "Capacity")
- $v_j$ = Value of product $j$ (from products.csv, column "Value")
- $w_j$ = Weight (space requirement) of product $j$ (from products.csv, column "Weight")

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each section $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
\]

For all $i \in S$, $j \in P$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Numerical Data:**

- Section capacities ($c_i$):

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

- Product values ($v_j$) and weights ($w_j$):

| ProductName | Value ($v_j$) | Weight ($w_j$) |
|-------------|--------------|---------------|
| 1           | 10           | 2             |
| 2           | 15           | 3             |
| 3           | 8            | 1             |
| 4           | 12           | 2             |
| 5           | 20           | 4             |
| 6           | 25           | 5             |
| 7           | 5            | 1             |
| 8           | 30           | 6             |
| 9           | 18           | 3             |
| 10          | 22           | 4             |

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i, \quad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,8,\ j = 1,\ldots,10
\end{align*}
\]

Where $v_j$ and $w_j$ are as given above, and $c_i$ is the section capacity as listed.
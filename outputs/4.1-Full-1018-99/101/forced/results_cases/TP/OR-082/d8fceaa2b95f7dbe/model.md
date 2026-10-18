Let $N = \{\text{Depot}, A, B, C\}$ be the set of locations (depot and three customers).

Define binary variables:
$$
x_{ij} =
\begin{cases}
1 & \text{if the van travels directly from location } i \text{ to location } j \\
0 & \text{otherwise}
\end{cases}
\quad \forall i, j \in N,\, i \neq j
$$

Let $d_{ij}$ be the distance from $i$ to $j$ (see table below).

Let $u_i$ be a continuous variable for subtour elimination, for $i \in \{A, B, C\}$.

Objective:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

Subject to:

1. Each location is departed from exactly once:
$$
\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
$$

2. Each location is arrived at exactly once:
$$
\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
$$

3. Subtour elimination (Miller-Tucker-Zemlin, for $i, j \in \{A, B, C\},\, i \neq j$):
$$
u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\, i \neq j
$$
with $1 \leq u_i \leq 3$ for $i \in \{A, B, C\}$.

4. Binary variables:
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
$$

#### Distance matrix (all values in km, extracted from DistanceMatrix.csv):

| From $\backslash$ To | Depot | A  | B  | C  |
|----------------------|-------|----|----|----|
| Depot                | 0     | 28 | 41 | 63 |
| A                    | 28    | 0  | 27 | 87 |
| B                    | 41    | 27 | 0  | 81 |
| C                    | 63    | 87 | 81 | 0  |

So, for example, $d_{\text{Depot},A} = 28$, $d_{A,B} = 27$, etc.

#### Full model:

Minimize
$$
28\,x_{\text{Depot},A} + 41\,x_{\text{Depot},B} + 63\,x_{\text{Depot},C}
+ 28\,x_{A,\text{Depot}} + 27\,x_{A,B} + 87\,x_{A,C}
+ 41\,x_{B,\text{Depot}} + 27\,x_{B,A} + 81\,x_{B,C}
+ 63\,x_{C,\text{Depot}} + 87\,x_{C,A} + 81\,x_{C,B}
$$

Subject to:
- $x_{\text{Depot},A} + x_{\text{Depot},B} + x_{\text{Depot},C} = 1$
- $x_{A,\text{Depot}} + x_{A,B} + x_{A,C} = 1$
- $x_{B,\text{Depot}} + x_{B,A} + x_{B,C} = 1$
- $x_{C,\text{Depot}} + x_{C,A} + x_{C,B} = 1$

- $x_{A,\text{Depot}} + x_{B,\text{Depot}} + x_{C,\text{Depot}} = 1$
- $x_{\text{Depot},A} + x_{B,A} + x_{C,A} = 1$
- $x_{\text{Depot},B} + x_{A,B} + x_{C,B} = 1$
- $x_{\text{Depot},C} + x_{A,C} + x_{B,C} = 1$

- $u_A - u_B + 3 x_{A,B} \leq 2$
- $u_A - u_C + 3 x_{A,C} \leq 2$
- $u_B - u_A + 3 x_{B,A} \leq 2$
- $u_B - u_C + 3 x_{B,C} \leq 2$
- $u_C - u_A + 3 x_{C,A} \leq 2$
- $u_C - u_B + 3 x_{C,B} \leq 2$

- $1 \leq u_A \leq 3$
- $1 \leq u_B \leq 3$
- $1 \leq u_C \leq 3$

- $x_{ij} \in \{0,1\}$ for all $i \neq j$, $i,j \in \{\text{Depot},A,B,C\}$

This model finds the minimum-distance tour starting and ending at the depot, visiting each of $A$, $B$, and $C$ exactly once.
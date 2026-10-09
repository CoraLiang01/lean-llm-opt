Let $N = \{\text{Depot}, A, B, C\}$ be the set of locations. Let $d_{ij}$ denote the distance from node $i$ to node $j$, for all $i, j \in N$, as given below.

Define binary decision variables:
$$
x_{ij} = \begin{cases}
1 & \text{if the van travels directly from node } i \text{ to node } j \\
0 & \text{otherwise}
\end{cases}
\quad \forall i, j \in N,\, i \neq j
$$

Let $u_i$ be a continuous variable for each $i \in N$, $i \neq \text{Depot}$, used for subtour elimination (Miller-Tucker-Zemlin formulation).

Objective:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

Subject to:

1. Each node is departed exactly once:
$$
\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
$$

2. Each node is arrived at exactly once:
$$
\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
$$

3. Subtour elimination (for all $i, j \in N$, $i \neq j$, $i \neq \text{Depot}$, $j \neq \text{Depot}$):
$$
u_i - u_j + 3 x_{ij} \leq 2
$$

4. Variable domains:
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j \\
u_i \in [1,3] \quad \forall i \in N,\, i \neq \text{Depot}
$$

Distance matrix (all values in km, as retrieved):

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C \\
\hline
\text{Depot} & 0 & 28 & 41 & 63 \\
A & 28 & 0 & 27 & 87 \\
B & 41 & 27 & 0 & 81 \\
C & 63 & 87 & 81 & 0 \\
\end{array}
\]

That is,
\[
\begin{aligned}
&d_{\text{Depot},A} = 28,\quad d_{\text{Depot},B} = 41,\quad d_{\text{Depot},C} = 63 \\
&d_{A,\text{Depot}} = 28,\quad d_{A,B} = 27,\quad d_{A,C} = 87 \\
&d_{B,\text{Depot}} = 41,\quad d_{B,A} = 27,\quad d_{B,C} = 81 \\
&d_{C,\text{Depot}} = 63,\quad d_{C,A} = 87,\quad d_{C,B} = 81 \\
\end{aligned}
\]

The optimal sequence of visits is the tour that minimizes the total travel distance, as determined by solving the above model.
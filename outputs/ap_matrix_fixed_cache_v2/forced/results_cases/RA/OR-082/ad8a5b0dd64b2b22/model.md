Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (locations). Let $d_{ij}$ denote the distance from node $i$ to node $j$, as given in the table below. Define binary variables $x_{ij}$, where $x_{ij} = 1$ if the route travels directly from node $i$ to node $j$, and $0$ otherwise, for all $i, j \in N$, $i \neq j$. Let $u_i$ be auxiliary variables for subtour elimination (for $i \in \{A, B, C\}$).

**Distance Matrix (retrieved data):**

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

**Mathematical Model:**

Minimize total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

Subject to:

1. **Each node is departed exactly once:**
$$
\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
$$

2. **Each node is arrived at exactly once:**
$$
\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
$$

3. **Subtour elimination (Miller-Tucker-Zemlin constraints):**
$$
u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\ i \neq j
$$
$$
1 \leq u_i \leq 3 \quad \forall i \in \{A, B, C\}
$$

4. **Variable domains:**
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\ i \neq j
$$

**Where:**
- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ are the distances as given above.

**Decision:**
The optimal sequence of visits is the tour corresponding to the values of $x_{ij}$ that minimize the objective.
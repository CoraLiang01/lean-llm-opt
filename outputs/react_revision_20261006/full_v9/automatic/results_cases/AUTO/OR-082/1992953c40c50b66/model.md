## Mathematical Model

Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (locations). Let $d_{ij}$ denote the distance from node $i$ to node $j$, as given in Data Mapping below.

**Decision variables:**
- $x_{ij} \in \{0,1\}$ for all $i, j \in N$, $i \neq j$: $x_{ij} = 1$ if the route goes directly from $i$ to $j$, $0$ otherwise.

**Parameters:**
- $d_{ij}$: distance from $i$ to $j$, from Data Mapping.

**Objective:**
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

**Constraints:**
1. **Each node (except Depot) is entered exactly once:**
   $$
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in \{A, B, C\}
   $$
2. **Each node (except Depot) is left exactly once:**
   $$
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in \{A, B, C\}
   $$
3. **Depot is left and entered exactly once:**
   $$
   \sum_{j \in N,\, j \neq \text{Depot}} x_{\text{Depot},j} = 1
   $$
   $$
   \sum_{i \in N,\, i \neq \text{Depot}} x_{i,\text{Depot}} = 1
   $$
4. **Subtour elimination (Miller-Tucker-Zemlin for 3 customers):**
   Introduce $u_i$ for $i \in \{A, B, C\}$, $1 \leq u_i \leq 3$:
   $$
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\, i \neq j
   $$

**Variable domains:**
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
$$
$$
u_i \in [1,3] \quad \forall i \in \{A, B, C\}
$$

---

### Data Mapping

- $N = \{\text{Depot}, A, B, C\}$
- $d_{ij}$ is the value in DistanceMatrix.csv, table_id: file_0_view_0, with row label $i$ (column "Unnamed: 0") and column $j$.

For example:
- $d_{\text{Depot},A} = 28$
- $d_{\text{Depot},B} = 41$
- $d_{\text{Depot},C} = 63$
- $d_{A,B} = 27$, $d_{A,C} = 87$, $d_{A,\text{Depot}} = 28$, etc.

Use all $d_{ij}$ for $i, j \in N$, $i \neq j$, as per the matrix.

---

**Summary:**  
This is a standard symmetric TSP for 4 nodes (Depot, A, B, C) with the distance matrix as above. The optimal sequence is the one that minimizes the total travel distance, found by solving the above model.
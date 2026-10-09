## Mathematical Model

**Sets**
- $N = \{\text{Depot}, A, B, C\}$: set of all locations (including depot and customers)
- $V = \{A, B, C\}$: set of customer locations

**Parameters** (from DistanceMatrix.csv, table_id: file_0_view_0)
- $d_{ij}$: distance from location $i$ to location $j$, for all $i, j \in N$

**Decision Variables**
- $x_{ij} \in \{0,1\}$: 1 if the route goes directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$
- $u_i \in [1, 3]$: auxiliary variables for subtour elimination, for all $i \in V$

**Objective**
Minimize total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

**Constraints**
1. **Depart from depot exactly once:**
$$
\sum_{\substack{j \in V}} x_{\text{Depot},j} = 1
$$

2. **Return to depot exactly once:**
$$
\sum_{\substack{i \in V}} x_{i,\text{Depot}} = 1
$$

3. **Each customer visited exactly once:**
$$
\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in V
$$
$$
\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in V
$$

4. **No subtours (MTZ constraints):**
$$
u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in V,\, i \neq j
$$
$$
1 \leq u_i \leq 3 \quad \forall i \in V
$$

5. **Variable domains:**
$$
x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
$$

**Data Mapping**
- $d_{ij}$ is given by the entry in DistanceMatrix.csv (table_id: file_0_view_0) at row $i$, column $j$, where $i, j \in N = \{\text{Depot}, A, B, C\}$.
- All sets and indices are defined by the unique location names in the matrix.

**Summary**
This is a standard TSP (Traveling Salesman Problem) for 4 nodes (Depot, A, B, C) with the objective to minimize total travel distance, using the pairwise distances from the provided matrix. The model ensures the van starts and ends at the depot, visits each customer exactly once, and prevents subtours.
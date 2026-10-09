## Mathematical Model

**Sets**
- $N = \{\text{Depot}, A, B, C\}$ (nodes; from DistanceMatrix.csv)
- $V = N \setminus \{\text{Depot}\} = \{A, B, C\}$ (customer locations)

**Parameters** (from DistanceMatrix.csv, table_id: file_0_view_0)
- $d_{ij}$: distance from node $i$ to node $j$, for all $i, j \in N$

**Decision Variables**
- $x_{ij} \in \{0,1\}$ 1 if the route goes directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$
- $u_i \in [1, |V|]$ subtour elimination variable for $i \in V$

**Objective**
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} \, x_{ij}
\]

**Constraints**
1. **Leave each node exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]
2. **Enter each node exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]
3. **No subtours (MTZ constraints):**
   \[
   u_i - u_j + 3\, x_{ij} \leq 2 \quad \forall i, j \in V,\, i \neq j
   \]
   \[
   1 \leq u_i \leq 3 \quad \forall i \in V
   \]
4. **No self-loops:**
   \[
   x_{ii} = 0 \quad \forall i \in N
   \]

**Variable Domains**
- $x_{ij} \in \{0,1\}$ for all $i, j \in N$, $i \neq j$
- $u_i \in [1,3]$ for all $i \in V$

**Data Mapping**
- $N$ and $d_{ij}$: from DistanceMatrix.csv (table_id: file_0_view_0, columns "Unnamed: 0" as row labels, "Depot", "A", "B", "C" as columns)
- $V = \{A, B, C\}$

**Interpretation**
- The optimal sequence is the order of $V$ corresponding to the $x_{ij}$ variables set to 1 in the optimal solution.
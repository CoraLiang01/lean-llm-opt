#### Abstract Mathematical Model: Symmetric Traveling Salesman Problem (TSP)

**Index Sets:**
- $N$: set of locations (customers), indexed by $i, j \in N$, with $|N| = 15$.

**Parameters:**
- $d_{ij}$: distance from location $i$ to location $j$, for all $i, j \in N$, $i \neq j$.

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: equals 1 if the tour travels directly from location $i$ to location $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in \mathbb{Z}$, $2 \leq u_i \leq |N|$, for all $i \in N \setminus \{1\}$: auxiliary variables for subtour elimination (Miller-Tucker-Zemlin formulation).

**Objective:**
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} \cdot x_{ij}
\]

**Constraints:**
1. **Departure Constraint:** Each location is departed exactly once:
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1, \quad \forall i \in N
   \]
2. **Arrival Constraint:** Each location is arrived at exactly once:
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1, \quad \forall j \in N
   \]
3. **Subtour Elimination (MTZ):** For all $i, j \in N \setminus \{1\}$, $i \neq j$:
   \[
   u_i - u_j + (|N| - 1) x_{ij} \leq |N| - 2
   \]
   with $u_1 = 1$.
4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in \mathbb{Z},\ 2 \leq u_i \leq |N|,\quad \forall i \in N \setminus \{1\}
   \]

**Data Mapping:**
- Table: 20.csv (table_id: file_0_view_0)
- Index set $N$ and parameter $d_{ij}$ are defined by the row and column headers in 20.csv.
- $d_{ij}$ is the value in column $j$ and row $i$ of 20.csv, for all $i \neq j$.

**Notes:**
- The tour starts and ends at location 1 (node 1).
- The distance matrix is symmetric: $d_{ij} = d_{ji}$.
- All locations must be visited exactly once.
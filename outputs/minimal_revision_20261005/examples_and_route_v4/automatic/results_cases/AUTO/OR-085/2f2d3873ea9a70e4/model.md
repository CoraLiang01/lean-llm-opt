**Abstract Mathematical Model**

**Sets:**
- $N$: set of locations/customers, indexed by $i, j$ (from 1 to 15).

**Parameters:**
- $d_{ij}$: distance from location $i$ to location $j$.

**Variables:**
- $x_{ij} \in \{0,1\}$: 1 if the tour goes directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in \mathbb{Z}_{\geq 0}$: auxiliary variables for subtour elimination, for all $i \in N$, $i \geq 2$.

**Objective:**
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

**Constraints:**

1. **Leave each location exactly once:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]

2. **Enter each location exactly once:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]

3. **Subtour elimination (Miller-Tucker-Zemlin, for $i, j \in N$, $i \neq j$, $i \geq 2$, $j \geq 2$):**
   \[
   u_i - u_j + 15 x_{ij} \leq 14 \quad \forall i, j \in N,\, i \neq j,\, i \geq 2,\, j \geq 2
   \]
   \[
   1 \leq u_i \leq 14 \quad \forall i \in N,\, i \geq 2
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in N,\, i \geq 2
   \]

**Data Mapping**

- $N$: All locations/customers, corresponding to the 15 rows and columns in 20.csv.
- $d_{ij}$: For $i, j \in N$, $i \neq j$, $d_{ij}$ is the value in 20.csv at row $i$, column $j$ (with both $i$ and $j$ as 1-based indices, matching the CSV's row and column labels).
  - Table: file_0_view_0
  - Row: $i$ (1-based, corresponds to source_row $i-1$)
  - Column: $j$ (column name as string, e.g., "2", "3", ..., "15")
- $x_{ij}$: Binary variable for each ordered pair $(i, j)$, $i \neq j$.
- $u_i$: Integer variable for each $i \in N$, $i \geq 2$.

**Notes:**
- The tour starts and ends at location 1.
- The distance matrix is symmetric; only the upper or lower triangle is filled in the CSV, but $d_{ij}$ is defined for all $i \neq j$ using the provided data.
- All indices and data mappings use the original file and column names as returned by CSVQA.
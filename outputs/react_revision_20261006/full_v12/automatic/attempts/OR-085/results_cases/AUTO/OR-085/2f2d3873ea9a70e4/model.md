## Mathematical Model

**Sets**
- $N = \{1,2,\ldots,15\}$: set of locations (with 1 as the depot/start/end)
- $d_{ij}$: distance from location $i$ to $j$, from 20.csv, for all $i,j \in N$, $i \neq j$

**Parameters**
- $d_{ij}$: symmetric distance between locations $i$ and $j$ (from Data Mapping below)

**Decision Variables**
- $x_{ij} \in \{0,1\}$: 1 if the route goes directly from $i$ to $j$, 0 otherwise, for all $i \neq j$, $i,j \in N$
- $u_i \in [2,15]$: position of node $i$ in the tour (for subtour elimination), for $i \in N$, $i \neq 1$

**Objective**
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
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
3. **Subtour elimination (MTZ):**
   \[
   u_i - u_j + 15\, x_{ij} \leq 14 \quad \forall i, j \in N,\, i \neq j,\, i \neq 1,\, j \neq 1
   \]
   \[
   2 \leq u_i \leq 15 \quad \forall i \in N,\, i \neq 1
   \]
4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in [2,15] \quad \forall i \in N,\, i \neq 1
   \]

**Data Mapping**
- $d_{ij}$: from 20.csv, table_id: file_0_view_0, columns: "2"–"15", rows: 0–14, where row $i-1$, column "$j$" gives $d_{ij}$ for $i < j$; for $i > j$, $d_{ij} = d_{ji}$ (symmetry); $d_{ii} = 0$.

**Notes**
- The tour starts and ends at location 1.
- All locations are visited exactly once.
- The MTZ constraints eliminate subtours.

**Summary**
Sets, parameters, variables, objective, and constraints are defined above. All data is mapped directly from the provided 20.csv file.
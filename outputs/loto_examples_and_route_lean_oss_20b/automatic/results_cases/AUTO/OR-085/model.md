##### Sets  

- \(N=\{1,2,\dots ,15\}\) (visit locations)  

##### Parameters  

- \(d_{ij}\) distance between location \(i\) and \(j\) (symmetric, \(d_{ij}=d_{ji}\)).  
  *Data mapping:*  
  \[
  d_{ij}= \text{value in row }(i-1),\text{ column }j \text{ of table } \texttt{file\_0\_view\_0}
  \]
  (row indices are 0‑based; column names are the node numbers 2–15; diagonal entries are 0).  

##### Decision Variables  

- \(x_{ij}\in\{0,1\}\) 1 if the tour goes directly from \(i\) to \(j\).  
- \(u_i\) continuous “position” variable for MTZ subtour elimination.  

##### Objective  

\[
\min \sum_{i\in N}\sum_{j\in N} d_{ij}\,x_{ij}
\]

##### Constraints  

1. **Out‑degree** \(\displaystyle \sum_{j\in N} x_{ij}=1 \quad \forall i\in N\)

2. **In‑degree** \(\displaystyle \sum_{i\in N} x_{ij}=1 \quad \forall j\in N\)

3. **MTZ subtour elimination** \(\displaystyle u_i - u_j + 15\,x_{ij}\le 14 \quad \forall i,j\in N,\; i\neq j,\; i\neq1,\; j\neq1\)

4. **Start position** \(u_1 = 1\)

5. **Bounds on \(u\)** \(2 \le u_i \le 15 \quad \forall i\in N,\; i\neq1\)

6. **Binary** \(x_{ij}\in\{0,1\}\quad \forall i,j\in N\)

##### Data Mapping Rules  

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(d_{ij}\) | Table **file_0_view_0** | For each row \(r\) (0‑based) and column name \(c\) (string of a node number), set \(i=r+1,\; j=\text{int}(c)\).  Assign \(d_{ij}\) the cell value.  Symmetry is enforced by \(d_{ji}=d_{ij}\).  Diagonal entries are 0. |

This model determines the optimal visiting order that starts and ends at location 1 while minimizing total travel distance.
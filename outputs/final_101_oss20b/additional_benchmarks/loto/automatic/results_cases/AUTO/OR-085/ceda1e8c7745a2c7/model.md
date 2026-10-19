**Mathematical formulation for the 15‑node symmetric TSP**

---

### Index sets  
- \(N=\{1,2,\dots ,15\}\) – set of all locations (node 1 is the depot).  
- \(E=\{(i,j)\mid i,j\in N,\; i<j\}\) – set of unordered edges.

### Parameters  

| Symbol | Meaning | Definition (using the table “file_0_view_0”) |
|--------|---------|----------------------------------------------|
| \(d_{ij}\) | distance between locations \(i\) and \(j\) | \[
d_{ij}= \begin{cases}
\text{value in row }0,\text{ column }j & \text{if } i=1,\\[4pt]
\text{value in row }(i-1),\text{ column }j & \text{if } i>1,\; j>i,\\[4pt]
\text{value in row }(j-1),\text{ column }i & \text{if } i>1,\; j<i,\\[4pt]
0 & \text{if } i=j .
\end{cases}
\]  
The table “file_0_view_0” contains the upper‑triangular part of the symmetric distance matrix in columns “2”–“15”.  Empty cells are interpreted as the corresponding lower‑triangular entry (by symmetry). |

### Decision variables  

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_{ij}\) | \(\{0,1\}\) for \((i,j)\in E\) | 1 if the tour uses the edge \((i,j)\); 0 otherwise. |
| \(u_i\) | \([2,\,N-1]\) for \(i\in N\setminus\{1\}\) | Continuous “position” variable used in the MTZ subtour‑elimination constraints. |
| \(u_1\) | fixed to 1 | Position of the depot. |

### Objective  

\[
\min \;\; \sum_{(i,j)\in E} d_{ij}\,x_{ij}
\]

(The sum is over all unordered edges; each edge is counted once because the matrix is symmetric.)

### Constraints  

1. **Degree constraints (each node has exactly two incident edges)**  

\[
\sum_{j<i} x_{ji} \;+\; \sum_{j>i} x_{ij} \;=\; 2
\qquad \forall\, i\in N
\]

2. **Subtour‑elimination (MTZ formulation)**  

\[
\begin{aligned}
&u_1 = 1,\\
&2 \;\le\; u_i \;\le\; N-1 \qquad \forall\, i\in N\setminus\{1\},\\[4pt]
&u_i - u_j \;+\; (N-1)\,x_{ij} \;\le\; N-2
\qquad \forall\, (i,j)\in E,\; i>1,\; j>1 .
\end{aligned}
\]

The MTZ constraints guarantee that no subtour can appear: if an edge \((i,j)\) is chosen (\(x_{ij}=1\)), the “positions” \(u_i\) and \(u_j\) must differ by at least 1, preventing a cycle that does not include the depot.

---

**Mapping to the data**

- The distance values are taken from the table **“file_0_view_0”**.  
- Column “2” corresponds to distances from node 1 to node 2, column “3” to node 1–3, …, column “15” to node 1–15.  
- For rows 2–15 (row index \(k-1\) for node \(k\)), the non‑empty entries in columns \(j>k\) give the distances \(d_{kj}\).  
- Because the matrix is symmetric, any missing entry in the lower‑triangular part is filled by the corresponding upper‑triangular value.

This formulation fully specifies the traveling salesman problem for the 15 locations described in the CSV file.
## Mathematical Model

**Sets**  
\(N=\{\text{Depot},A,B,C\}\)  \(C=N\setminus\{\text{Depot}\}\)

**Parameters**  
\(d_{ij}\) – road distance (km) from node \(i\) to node \(j\)  
 \(d_{ij}\) is taken from *DistanceMatrix.csv* (table_id `file_0_view_0`).

**Decision variables**  
\(x_{ij}\in\{0,1\}\)  for all \(i\in N,\;j\in N,\;i\neq j\)  
 \(x_{ij}=1\) if the van travels directly from \(i\) to \(j\).

\(u_i\in[1,|C|]\)  for all \(i\in C\)  
 \(u_i\) is an auxiliary position variable used in the MTZ subtour‑elimination constraints.

---

### Objective
\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

### Constraints

1. **Out‑degree** – each node has exactly one outgoing edge  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1
   \qquad \forall\, i\in N
   \]

2. **In‑degree** – each node has exactly one incoming edge  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1
   \qquad \forall\, j\in N
   \]

3. **MTZ subtour elimination** – for all customer pairs \(i,j\in C,\;i\neq j\)  
   \[
   u_i - u_j + (|C|)\,x_{ij}\;\le\;|C|-1
   \qquad \forall\, i,j\in C,\;i\neq j
   \]

   (Here \(|C|=3\), so the right‑hand side is \(2\).)

4. **Variable domains**  
   \[
   x_{ij}\in\{0,1\}\quad\forall\,i\neq j,\qquad
   u_i\in[1,3]\quad\forall\,i\in C
   \]

---

## Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(d_{ij}\) | *DistanceMatrix.csv* (table_id `file_0_view_0`) | For each row where `Unnamed: 0` = \(i\) and each column header \(j\) (excluding `Unnamed: 0`), the cell value is \(d_{ij}\). |
| \(N\) | Implicit from the set of distinct node names in the matrix | `{Depot, A, B, C}` |
| \(C\) | \(N\setminus\{\text{Depot}\}\) | `{A, B, C}` |

No additional parameters are required. The model uses only the distances supplied in the matrix and the standard TSP structure.
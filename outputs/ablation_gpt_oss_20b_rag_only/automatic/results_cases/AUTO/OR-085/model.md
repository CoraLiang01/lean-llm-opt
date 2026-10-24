**Mathematical Model**

Let  

* \(N=\{1,\dots ,15\}\) – set of customer locations (node 1 is the depot).  
* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the tour uses the directed arc from node \(i\) to node \(j\) (\(i\neq j\)).  
* \(u_i\in\{1,\dots ,14\}\) – integer “position” of node \(i\) in the tour (only for \(i\neq1\)).  

The distance between two distinct nodes is denoted by the parameter \(d_{ij}\) (symmetric, \(d_{ij}=d_{ji}\), \(d_{ii}=0\)).  

---

**Objective**

\[
\min \;\; \sum_{\substack{i\in N\\ j\in N\\ i\neq j}} d_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Degree constraints (Hamiltonian cycle)**  

   *Depot (node 1) has exactly one outgoing and one incoming arc*  

   \[
   \sum_{j\in N,\;j\neq1} x_{1j}=1,\qquad
   \sum_{i\in N,\;i\neq1} x_{i1}=1
   \]

   *Every other node has exactly one outgoing and one incoming arc*  

   \[
   \sum_{j\in N,\;j\neq i} x_{ij}=1,\qquad
   \sum_{i\in N,\;i\neq j} x_{ij}=1
   \quad\forall\, i\in N\setminus\{1\}
   \]

2. **Subtour‑elimination (MTZ)**  

   For all distinct non‑depot nodes \(i,j\in N\setminus\{1\}\):

   \[
   u_i - u_j + 14\,x_{ij}\;\le\;13
   \]

   (Here \(n-1=14\) and the big‑\(M\) is \(n-2=13\).)

3. **Variable domains**

   \[
   x_{ij}\in\{0,1\}\quad\forall\, i\neq j,\qquad
   u_i\in\{1,\dots ,14\}\quad\forall\, i\neq1
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | Table **file_0_view_0** (CSV 20) | For each pair \(i<j\) (with \(i,j\in N\)):<br>• Row index = \(i-1\) (0‑based).<br>• Column name = string representation of \(j\).<br>• Value = numeric entry in that cell (if present).<br>For \(i>j\), set \(d_{ij}=d_{ji}\).<br>For \(i=j\), set \(d_{ii}=0\). |
| \(N\) | Implicit | \(N=\{1,\dots ,15\}\) (all rows returned). |
| \(x_{ij}\) | Decision variable | Binary variable for each ordered pair \(i\neq j\). |
| \(u_i\) | Decision variable | Integer variable for each \(i\neq1\). |

*Notes on the distance data:*  
The CSV contains only the upper‑triangular part of the symmetric distance matrix (row 0 holds distances from node 1 to nodes 2–15, row 1 holds distances from node 2 to nodes 3–15, etc.). Empty cells correspond to diagonal entries or omitted lower‑triangular entries and are ignored. The mapping rule above reconstructs the full symmetric matrix needed for the objective.
## Mathematical Model

**Sets**

- \(N=\{0,1,2,3\}\) (0 = Depot, 1 = A, 2 = B, 3 = C)  
- \(C=\{1,2,3\}\) (customers)

**Parameters**

- \(d_{ij}\) – road distance (km) between node \(i\) and node \(j\)  
  (obtained from *DistanceMatrix.csv*; see Data Mapping)

**Decision Variables**

- \(x_{ij}\in\{0,1\}\) = 1 if the van travels directly from node \(i\) to node \(j\)  
- \(u_i\in[1,|C|]\) auxiliary position variable for customer \(i\) (MTZ)

**Objective**

\[
\min \;\sum_{i\in N}\sum_{j\in N,\;j\neq i} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Depot departure/arrival**

\[
\sum_{j\in C}x_{0j}=1,\qquad
\sum_{i\in C}x_{i0}=1
\]

2. **Customer visit**

\[
\sum_{j\in N,\;j\neq i}x_{ij}=1,\qquad
\sum_{j\in N,\;j\neq i}x_{ji}=1
\quad\forall i\in C
\]

3. **Sub‑tour elimination (MTZ)**  

\[
u_i-u_j+|C|\,x_{ij}\le |C|-1
\quad\forall i,j\in C,\;i\neq j
\]

4. **Variable bounds**

\[
x_{ij}\in\{0,1\}\;\forall i\neq j,\qquad
1\le u_i\le |C|\;\forall i\in C
\]

---

## Data Mapping

| File Path | Parameter | Source Column | Source Row | Notes |
|-----------|-----------|---------------|------------|-------|
| `Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv` | \(d_{ij}\) | Column *i*, Row *j* | All rows/columns for nodes 0 (Depot), 1 (A), 2 (B), 3 (C) | Distances are symmetric; only the upper‑triangle values are needed for the objective. |

*The model uses only the distances between the depot and the three customers, and between the customers themselves, as extracted from the specified CSV file.*
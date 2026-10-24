## Mathematical Model

**Sets**

- \(N=\{1,2,\dots ,15\}\) – customers (node 1 is the depot).

**Parameters**

- \(d_{ij}\) – distance between node \(i\) and node \(j\)  
  (symmetric, \(d_{ij}=d_{ji}\), obtained from *20.csv*).

**Decision Variables**

- \(x_{ij}\in\{0,1\}\) for all \(i,j\in N,\; i\neq j\)  
  (1 if the tour goes directly from \(i\) to \(j\), 0 otherwise).

- \(u_i\) for all \(i\in N\setminus\{1\}\)  
  (continuous “position” variables used for subtour elimination).

**Objective**

\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Degree constraints (one incoming and one outgoing edge per node)**  

\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \quad \forall i\in N
\]
\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ji}=1 \quad \forall i\in N
\]

2. **Subtour elimination (MTZ formulation)**  

\[
u_i - u_j + 15\,x_{ij} \le 14 \quad \forall i,j\in N,\; i\neq j,\; i,j\neq 1
\]
\[
2 \le u_i \le 15 \quad \forall i\in N\setminus\{1\}
\]
\[
u_1 = 1
\]

The MTZ constraints guarantee that the tour is a single Hamiltonian cycle that starts and ends at node 1.

---

## Data Mapping

| File Path | Parameter | Mapping Rule |
|-----------|-----------|--------------|
| `Test_Dataset/Large-scale-or/Others_example/Others7/20.csv` | \(d_{ij}\) | For each row \(i\) (column *Unnamed: 0*) and each column \(j\) (columns 1–15), set \(d_{ij}\) to the numeric value in that cell. The matrix is symmetric; only the upper‑triangular part is needed for the model. |

No other data sources are required. The model uses only the distance matrix and the standard TSP formulation with MTZ subtour elimination.
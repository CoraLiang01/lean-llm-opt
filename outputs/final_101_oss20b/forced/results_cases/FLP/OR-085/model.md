**Symbolic Model**

Let  

* \(N=\{1,\dots ,15\}\) – set of locations (node 1 is the depot).  
* \(d_{ij}\) – distance between location \(i\) and \(j\) (symmetric, \(d_{ii}=0\)).  

Decision variables  

\[
x_{ij}\in\{0,1\}\quad\forall i,j\in N,\; i\neq j
\]
\[
u_i\in\mathbb{Z}\quad\forall i\in N\setminus\{1\}
\]

Objective  

\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

Constraints  

1. **One outgoing arc per node**  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \qquad\forall i\in N
   \]

2. **One incoming arc per node**  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \qquad\forall j\in N
   \]

3. **MTZ subtour elimination** (for all non‑depot nodes)  
   \[
   u_i - u_j + 15\,x_{ij}\le 14 \qquad\forall i,j\in N\setminus\{1\},\; i\neq j
   \]

4. **Fix depot rank**  
   \[
   u_1 = 1
   \]

Domain  

\[
x_{ij}\in\{0,1\},\qquad u_i\in\{2,\dots ,15\}\;(i\neq 1)
\]

---

**Data Mapping Rules**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` (table id `file_0_view_0`) | For \(i=1\) and \(j>1\): <br> \(d_{1j}\) = value in row 0, column `j`. <br>For \(i>1\) and \(j>i\): <br> \(d_{ij}\) = value in row \(i-1\), column `j`. <br>For \(i>j\): <br> \(d_{ij}=d_{ji}\). <br>All other entries are omitted (diagonal \(d_{ii}=0\)). |

*The table `file_0_view_0` contains 15 rows (data_row_number 0–14) and columns labeled `"2"`–`"15"`.  
Row 0 corresponds to node 1, row 1 to node 2, …, row 14 to node 15.*
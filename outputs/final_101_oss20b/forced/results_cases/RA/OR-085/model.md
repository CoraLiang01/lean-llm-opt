**Mathematical Model (TSP with 15 customers)**  

Let  

* \(N=\{1,\dots ,15\}\) – set of locations (customer 1 is the depot).  
* \(d_{ij}\) – distance between location \(i\) and \(j\) (symmetric, \(d_{ii}=0\)).  
* \(x_{ij}\in\{0,1\}\) – binary variable, 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\) – continuous “order” variable for MTZ subtour elimination (\(u_1=1\)).  

---

### Objective  

\[
\min \;\; \sum_{i=1}^{15}\sum_{\substack{j=1\\ j\neq i}}^{15} d_{ij}\,x_{ij}\; .
\]

(Using symmetry, the sum can be taken over \(i<j\) with \(x_{ij}+x_{ji}\).)

---

### Constraints  

1. **Out‑degree (exactly one outgoing edge)**  

\[
\sum_{\substack{j=1\\ j\neq i}}^{15} x_{ij}=1 ,\qquad \forall i\in N .
\]

2. **In‑degree (exactly one incoming edge)**  

\[
\sum_{\substack{j=1\\ j\neq i}}^{15} x_{ji}=1 ,\qquad \forall i\in N .
\]

3. **MTZ Subtour elimination**  

\[
\begin{aligned}
&u_1 = 1,\\
&2 \le u_i \le 15 ,\qquad \forall i\in N\setminus\{1\},\\
&u_i - u_j + 14\,x_{ij} \le 13 ,\qquad \forall i,j\in N\setminus\{1\},\; i\neq j .
\end{aligned}
\]

4. **Variable domains**  

\[
x_{ij}\in\{0,1\}\;\; (i\neq j), \qquad u_i\in\mathbb{R}\;\; (i\neq 1).
\]

---

## Data Mapping

| Parameter | Index Set | Source | Mapping Rule |
|-----------|-----------|--------|--------------|
| \(d_{ij}\) | \(i,j\in N,\; i<j\) | `file_0_view_0` | For each table row with `data_row_number = r` (0‑based), let \(i=r+1\). For each column `j` (string “2”–“15”), if the cell value is present, set \(d_{ij}=\text{value}\). Missing entries are treated as 0. Symmetry: \(d_{ji}=d_{ij}\). |
| \(x_{ij}\) | \(i,j\in N,\; i\neq j\) | – | Decision variable. |
| \(u_i\) | \(i\in N\setminus\{1\}\) | – | Decision variable. |

*The table `file_0_view_0` contains the upper‑triangular distance matrix; each row corresponds to a source location \(i\) (row 0 → location 1, row 1 → location 2, …). Columns are destination locations \(j\) (strings “2”–“15”).*

---

**Note**: The model uses the MTZ formulation for subtour elimination. All constraints and the objective are expressed symbolically; numerical values are supplied via the data mapping above.
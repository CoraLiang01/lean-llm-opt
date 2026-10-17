**Mathematical Model (TSP with 15 customers, start‑and‑end at node 1)**  

Let  

* \(N=\{1,2,\dots ,15\}\) – set of locations (node 1 is the depot).  
* \(d_{ij}\) – distance between location \(i\) and \(j\) (symmetric, \(d_{ii}=0\)).  
* \(x_{ij}\in\{0,1\}\) – binary variable, 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\) – auxiliary continuous variable for MTZ subtour elimination (\(u_1=1\), \(2\le u_i\le 15\) for \(i>1\)).  

---

### Objective  

\[
\min \;\; \sum_{i\in N}\sum_{j\in N} d_{ij}\,x_{ij}
\]

---

### Constraints  

1. **Out‑degree (each node leaves exactly one edge)**  

\[
\sum_{j\in N} x_{ij}=1 \quad \forall i\in N
\]

2. **In‑degree (each node enters exactly one edge)**  

\[
\sum_{i\in N} x_{ij}=1 \quad \forall j\in N
\]

3. **Subtour elimination (MTZ)**  

\[
u_i - u_j + 15\,x_{ij} \le 14 \quad \forall i\in N,\; j\in N,\; i\neq j,\; i>1,\; j>1
\]

4. **Variable domains**  

\[
x_{ij}\in\{0,1\}\quad \forall i,j\in N
\]
\[
u_1=1,\qquad 2\le u_i\le 15 \quad \forall i>1
\]

---

## Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | **table_id**: `file_0_view_0` <br> **columns**: `2,3,4,5,6,7,8,9,10,11,12,13,14,15` | For each record (row index \(r\), 0‑based) and each column \(c\):<br>• If the cell value is non‑empty, set \(d_{(r+1),c}= \text{value}\).<br>• Set \(d_{c,(r+1)}= \text{value}\) (symmetry).<br>• For all \(i\), set \(d_{ii}=0\).<br>• If a cell is empty, treat \(d_{ij}=0\) (distance not specified). |
| \(x_{ij}\) | Decision variable | No external data. |
| \(u_i\) | Decision variable | No external data. |

*All other parameters are derived from the data mapping above.*
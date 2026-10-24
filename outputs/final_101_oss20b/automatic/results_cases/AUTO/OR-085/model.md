## Mathematical Model

**Sets**  
\(N=\{1,2,\dots ,15\}\) (cities, with 1 the depot)

**Parameters**  
\(d_{ij}\) – distance between city \(i\) and city \(j\) (symmetric, \(d_{ij}=d_{ji}\))

**Decision variables**  
\(x_{ij}\in\{0,1\}\) =1 if the tour goes directly from \(i\) to \(j\) (\(i\neq j\))  
\(u_i\) continuous, \(1\le u_i\le 14\) for \(i=2,\dots ,15\) (used for MTZ subtour elimination)  
\(u_1=0\) (fixed)

**Objective**  
\[
\min \sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Out‑degree** (one outgoing edge from each city)  
\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \qquad \forall i\in N
\]

2. **In‑degree** (one incoming edge to each city)  
\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ji}=1 \qquad \forall i\in N
\]

3. **MTZ subtour elimination**  
\[
u_i - u_j + 14\,x_{ij} \le 13 \qquad \forall i,j\in N,\; i\neq j,\; i\neq 1,\; j\neq 1
\]

4. **Variable domains**  
\[
x_{ij}\in\{0,1\}\quad (i\neq j),\qquad
u_i\in[1,14]\ (i=2,\dots ,15),\qquad u_1=0
\]

---

## Data Mapping

| Table ID | Column | Mapping Rule |
|----------|--------|--------------|
| `file_0_view_0` | `2`–`15` | For each record with `data_row_number = r` (0‑based), let `i = r+1`. For each column `c` (2–15) with a non‑empty value `v`, set: <br>• `d_{i,c} = v` <br>• `d_{c,i} = v` (ensuring symmetry) |

All other entries in the matrix are inferred by symmetry; no additional parameters are required. The index set \(N\) is derived from the 15 rows of the table. No other data are used.
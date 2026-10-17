## Mathematical Model

**Sets**  
\(N=\{1,2,\dots ,15\}\)

**Parameters**  
\(d_{ij}\) – distance between location \(i\) and location \(j\)  
(see *Data Mapping* below)

**Decision variables**  
\[
x_{ij}\in\{0,1\}\quad \forall i,j\in N,\; i\neq j
\]
\[
u_i\in\mathbb{Z}\quad \forall i\in\{2,\dots ,15\}
\]

**Objective**  
\[
\min \;\sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Degree constraints** – each node has exactly one outgoing and one incoming edge  
\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \quad \forall i\in N
\]
\[
\sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \quad \forall j\in N
\]

2. **Subtour elimination (MTZ)** – no subtours among nodes \(2\)–\(15\)  
\[
u_i - u_j + 14\,x_{ij}\le 13 \quad \forall i,j\in\{2,\dots ,15\},\; i\neq j
\]

3. **Bounds on order variables**  
\[
1 \le u_i \le 14 \quad \forall i\in\{2,\dots ,15\}
\]

---

## Data Mapping

| Table ID | Column | Source Row | Mapping |
|----------|--------|------------|---------|
| `file_0_view_0` | `2`–`15` | `0`–`14` | For each source row \(r\) (0‑based) let \(i=r+1\).  
For each column header `c` (string `"2"`…`"15"`) let \(j=\text{int}(c)\).  
If \(i<j\) then \(d_{ij}\) is the numeric value in that cell.  
If \(i>j\) then \(d_{ij}=d_{ji}\) (symmetry). |
| | | | All distances are positive integers; missing lower‑triangle cells are filled by symmetry. |

The parameter set \(d_{ij}\) is thus fully determined from the 20.csv file. No other data are required.
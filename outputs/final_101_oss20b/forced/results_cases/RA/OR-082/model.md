**Mathematical Model**

Let  

\[
N=\{\text{Depot},A,B,C\}
\]

\[
A=\{(i,j)\mid i\in N,\;j\in N,\;i\neq j\}
\]

\[
S=\{(i,j)\mid i\in\{A,B,C\},\;j\in\{A,B,C\},\;i\neq j\}
\]

Decision variables  

\[
x_{ij}\in\{0,1\}\qquad (i,j)\in A
\]

\[
u_i\in\{2,3\}\qquad i\in\{A,B,C\}
\]

Parameters  

\[
d_{ij}\quad\text{(distance from }i\text{ to }j\text{)}\qquad (i,j)\in A
\]

---

### Objective  

\[
\min \;\sum_{(i,j)\in A} d_{ij}\,x_{ij}
\]

### Constraints  

1. **Out‑degree 1**  

\[
\sum_{j\in N,\;j\neq i} x_{ij}=1 \qquad \forall i\in N
\]

2. **In‑degree 1**  

\[
\sum_{i\in N,\;i\neq j} x_{ij}=1 \qquad \forall j\in N
\]

3. **Subtour elimination (MTZ)**  

\[
u_i - u_j + 3\,x_{ij}\le 2 \qquad \forall (i,j)\in S
\]

4. **Bounds on \(u\)**  

\[
2 \le u_i \le 3 \qquad \forall i\in\{A,B,C\}
\]

5. **Binary variables**  

\[
x_{ij}\in\{0,1\}\qquad \forall (i,j)\in A
\]

---

## Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` | For each \((i,j)\in A\) with \(i\neq j\):<br>`d_{ij} = table_0_view_0[ row where Unnamed:0 = i ][ column j ]` |
| Node set \(N\) | `file_0_view_0` | `N = {"Depot","A","B","C"}` (extracted from column headers and the `Unnamed: 0` column) |

All other data (e.g., diagonal zeros) are not used in the model.
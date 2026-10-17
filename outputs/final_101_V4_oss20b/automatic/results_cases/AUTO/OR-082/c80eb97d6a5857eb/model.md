## Mathematical Model

**Sets**

\[
N=\{\text{Depot},A,B,C\}
\]

**Parameters**

\[
d_{ij}\;=\;\text{distance from node }i\text{ to node }j
\quad\forall i,j\in N,\;i\neq j
\]

**Decision Variables**

\[
x_{ij}\in\{0,1\}\quad\forall i,j\in N,\;i\neq j
\]
\[
u_i\in[1,|N|-1]\quad\forall i\in\{A,B,C\}
\]

**Objective**

\[
\min \sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Out‑degree 1**

\[
\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1
\quad\forall i\in N
\]

2. **In‑degree 1**

\[
\sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1
\quad\forall j\in N
\]

3. **MTZ Sub‑tour elimination**

\[
u_i-u_j+(|N|-1)\,x_{ij}\le |N|-2
\quad\forall i,j\in\{A,B,C\},\;i\neq j
\]

4. **Bounds on \(u\)**  

\[
1\le u_i\le |N|-1
\quad\forall i\in\{A,B,C\}
\]

5. **Binary and continuous domains**

\[
x_{ij}\in\{0,1\}\quad\forall i\neq j,\qquad
u_i\in\mathbb{R}\quad\forall i\in\{A,B,C\}
\]

---

## Data Mapping

| Table ID | Column | Description | Mapping Rule |
|----------|--------|-------------|--------------|
| `file_0_view_0` | `Unnamed: 0` | Node label (row) | `i` in \(d_{ij}\) |
| `file_0_view_0` | `Depot`, `A`, `B`, `C` | Distance to column node | `j` in \(d_{ij}\) |
| | | | \(d_{ij}\) = value in row where `Unnamed: 0` = \(i\) and column = \(j\) |

The numeric values for \(d_{ij}\) are taken directly from the four rows of the table:

- \(d_{\text{Depot},A}=28,\; d_{\text{Depot},B}=41,\; d_{\text{Depot},C}=63\)
- \(d_{A,\text{Depot}}=28,\; d_{A,B}=27,\; d_{A,C}=87\)
- \(d_{B,\text{Depot}}=41,\; d_{B,A}=27,\; d_{B,C}=81\)
- \(d_{C,\text{Depot}}=63,\; d_{C,A}=87,\; d_{C,B}=81\)

All other \(d_{ii}\) are omitted (distance from a node to itself is not used).
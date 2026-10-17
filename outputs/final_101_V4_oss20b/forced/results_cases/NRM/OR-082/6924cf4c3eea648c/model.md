**Symbolic Model**

| Symbol | Meaning |
|--------|---------|
| \(N\) | \(\{Depot, A, B, C\}\) |
| \(C\) | \(N\setminus\{Depot\}=\{A,B,C\}\) |
| \(d_{ij}\) | Distance from node \(i\) to node \(j\) (parameter) |
| \(x_{ij}\) | Binary decision: 1 if the route goes directly from \(i\) to \(j\) |
| \(u_i\) | Continuous “position” variable for node \(i\in C\) (MTZ) |

**Parameters**

\[
d_{ij}\quad \forall i,j\in N,\; i\neq j
\]

**Variables**

\[
x_{ij}\in\{0,1\}\quad \forall i,j\in N,\; i\neq j
\]
\[
u_i\in\{1,2,3\}\quad \forall i\in C
\]

**Objective**

\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Depot departure / arrival**

\[
\sum_{\substack{j\in N\\ j\neq Depot}} x_{Depot,j}=1
\]
\[
\sum_{\substack{i\in N\\ i\neq Depot}} x_{i,Depot}=1
\]

2. **Customer visit (exactly one in‑ and one out‑going arc)**

\[
\sum_{\substack{j\in N\\ j\neq i}} x_{i,j}=1 \quad \forall i\in C
\]
\[
\sum_{\substack{i\in N\\ i\neq j}} x_{i,j}=1 \quad \forall j\in C
\]

3. **MTZ subtour elimination**

\[
u_i - u_j + 3\,x_{ij}\;\le\;2 \quad \forall i,j\in C,\; i\neq j
\]

4. **Variable domains**

\[
x_{ij}\in\{0,1\}\quad \forall i\neq j
\]
\[
u_i\in\{1,2,3\}\quad \forall i\in C
\]

---

### Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | **table_id**: `file_0_view_0` | For each pair \((i,j)\) with \(i\neq j\): <br>• Locate the row where column **`Unnamed: 0`** equals \(i\). <br>• Read the cell in column \(j\). <br>• Assign that numeric value to \(d_{ij}\). |

*The table `file_0_view_0` contains the 4×4 distance matrix with rows and columns labeled `Depot`, `A`, `B`, `C`. No other data sources are used.*
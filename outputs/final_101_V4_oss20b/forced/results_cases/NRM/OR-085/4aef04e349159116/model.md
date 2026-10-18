**Abstract Model**

**Index Set**  
\(N=\{1,2,\dots ,15\}\)

**Parameters**  
\(d_{ij}\) – distance between location \(i\) and location \(j\) for all \(i,j\in N,\;i\neq j\).  
\(d_{ij}\) is obtained from the table **file_0_view_0**: for \(j>1\) it is the value in column named \(\text{``}j\text{''}\) of row \(i\); for \(j=1\) the distance is symmetric, i.e. \(d_{i1}=d_{1i}\).

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) for all \(i,j\in N,\;i\neq j\) (1 if the tour goes directly from \(i\) to \(j\), 0 otherwise)  
\(u_i\in\mathbb{R}\) for all \(i\in N\setminus\{1\}\) (continuous auxiliary variables for subtour elimination)

**Objective Function**  
\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Out‑degree and In‑degree Constraints**  
   \[
   \sum_{\substack{j\in N\\ j\neq 1}} x_{1j}=1,\qquad
   \sum_{\substack{i\in N\\ i\neq 1}} x_{i1}=1
   \]
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1,\qquad
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1
   \quad\forall\, i\in N\setminus\{1\}
   \]

2. **Subtour Elimination (MTZ) Constraints**  
   \[
   u_i - u_j + (|N|-1)\,x_{ij}\;\le\;|N|-2
   \quad\forall\, i,j\in N\setminus\{1\},\; i\neq j
   \]
   \[
   1 \;\le\; u_i \;\le\; |N|-1
   \quad\forall\, i\in N\setminus\{1\}
   \]
   \[
   u_1 = 0
   \]

3. **Variable Domains**  
   \[
   x_{ij}\in\{0,1\}\quad\forall\, i,j\in N,\;i\neq j
   \]
   \[
   u_i\in\mathbb{R}\quad\forall\, i\in N\setminus\{1\}
   \]

---

**Data Mapping**

| Parameter | Source Table | Column Names |
|-----------|--------------|--------------|
| \(d_{ij}\) | **file_0_view_0** | `"2"`, `"3"`, `"4"`, `"5"`, `"6"`, `"7"`, `"8"`, `"9"`, `"10"`, `"11"`, `"12"`, `"13"`, `"14"`, `"15"` |

The table **file_0_view_0** contains the distance values; each row corresponds to a source location \(i\) and each column (named `"2"`–`"15"`) gives the distance to the destination location \(j\) (for \(j>1\)). Distances to location 1 are obtained by symmetry. No record values or counts are reproduced here.
**Mathematical Model**

Let  

* \(I=\{Depot,\;A,\;B,\;C\}\) – set of nodes.  
* \(d_{ij}\) – road distance from node \(i\) to node \(j\).  
* \(x_{ij}\in\{0,1\}\) – binary variable, 1 if the van travels directly from \(i\) to \(j\).  
* \(u_i\in\{1,2\}\) – auxiliary position variable for \(i\in\{A,B,C\}\).

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in I,\;j\neq i} d_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{j\in I,\;j\neq i} x_{ij}=1 &&\forall i\in I \quad\text{(outgoing)}\\
&\sum_{i\in I,\;i\neq j} x_{ij}=1 &&\forall j\in I \quad\text{(incoming)}\\[4pt]
&u_i - u_j + 3\,x_{ij}\le 2 &&\forall i,j\in\{A,B,C\},\; i\neq j\\
&u_i\in\{1,2\} &&\forall i\in\{A,B,C\}\\
&x_{ij}\in\{0,1\} &&\forall i,j\in I,\; i\neq j
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source | Mapping |
|----------------------|--------|---------|
| \(I\) | `file_0_view_0` | Node identifiers: row names and column names (`Depot`, `A`, `B`, `C`). |
| \(d_{ij}\) | `file_0_view_0` | For each row `i` and column `j` (excluding the `Unnamed: 0` column), map the numeric value to \(d_{ij}\). |
| \(x_{ij}\) | — | Binary decision variable for each ordered pair \(i\neq j\) in \(I\). |
| \(u_i\) | — | Integer auxiliary variable for each \(i\in\{A,B,C\}\). |

*Table ID:* `file_0_view_0`  
*Columns used:* `Depot`, `A`, `B`, `C` (distance values).  
*Rows used:* `Depot`, `A`, `B`, `C` (as row identifiers).  

All numeric values from the distance matrix are taken verbatim; no aggregation or transformation is applied.
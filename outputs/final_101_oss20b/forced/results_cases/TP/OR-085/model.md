**Mathematical Model**

Let  

* \(N=\{1,2,\dots ,15\}\) – set of locations (node 1 is the depot).  
* \(d_{ij}\) – distance between locations \(i\) and \(j\) (symmetric, \(d_{ii}=0\)).  
* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\) – auxiliary position variable for node \(i\) (\(u_1=0\), \(1\le u_i\le 14\) for \(i\neq1\)).  

\[
\begin{aligned}
\min\;&\sum_{\substack{i,j\in N\\ i\neq j}} d_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 &&\forall i\in N \quad\text{(out‑degree)}\\
&\sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 &&\forall j\in N \quad\text{(in‑degree)}\\
&u_i-u_j+14\,x_{ij}\le 13 &&\forall i,j\in N,\; i\neq j,\; i,j\neq1 \quad\text{(MTZ)}\\
&u_1=0,\qquad 1\le u_i\le 14 &&\forall i\in N,\; i\neq1\\
&x_{ij}\in\{0,1\} &&\forall i,j\in N,\; i\neq j
\end{aligned}
\]

The constraints guarantee a single Hamiltonian cycle that starts and ends at node 1, and the objective minimizes the total travel distance.

---

**Data Mapping Rules**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(N\) | Implicit | \(N=\{1,\dots ,15\}\) |
| \(d_{ij}\) | Table **file_0_view_0** | For each pair \((i,j)\) with \(i\neq j\):<br>• If the cell at **row \(i-1\)**, **column \(j\)** contains a numeric value \(v\), set \(d_{ij}=v\).<br>• Else if the cell at **row \(j-1\)**, **column \(i\)** contains a numeric value \(v\), set \(d_{ij}=v\).<br>• Else set \(d_{ij}=0\).<br>For \(i=j\), set \(d_{ii}=0\). |
| \(x_{ij}\) | Decision variable | No external data; binary variable defined in the model. |
| \(u_i\) | Decision variable | No external data; integer variable defined in the model. |

*Table columns are named by the strings `"2"`, `"3"`, …, `"15"`.  
Row 0 corresponds to node 1, row 1 to node 2, …, row 14 to node 15.*
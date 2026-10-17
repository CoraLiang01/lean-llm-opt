**Mathematical model**

Let  

* \(N=\{1,2,\dots ,15\}\) – set of locations (node 1 is the depot/start).  
* \(d_{ij}\) – distance from location \(i\) to location \(j\) (symmetric, \(d_{ii}=0\)).  
* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\) – continuous “position” variable used for subtour elimination (\(u_1=1\)).

\[
\begin{aligned}
\min\;&\sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 &&\forall i\in N \quad\text{(one outgoing)}\\
&\sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 &&\forall j\in N \quad\text{(one incoming)}\\[4pt]
&u_i-u_j+15\,x_{ij}\le 14 &&\forall i,j\in N,\; i\neq j,\; i\neq 1,\; j\neq 1\\
&u_1=1,\qquad 2\le u_i\le 15 &&\forall i\in N,\; i\neq 1\\[4pt]
&x_{ij}\in\{0,1\} &&\forall i,j\in N,\; i\neq j
\end{aligned}
\]

The objective minimizes the total traveling distance while the constraints enforce a Hamiltonian cycle that starts and ends at node 1.

---

**Data‑mapping rules**

| Parameter | Source table | Row mapping | Column mapping | Notes |
|-----------|--------------|-------------|----------------|-------|
| \(d_{ij}\) | `file_0_view_0` | Row \(i-1\) (0‑based) | Column named `"j"` (string `"2"` … `"15"`) | Distances are symmetric; missing entries are treated as zero for \(i=j\). |

All other data (sets, variable domains, objective sense) are defined by the model above and do not require additional tables.
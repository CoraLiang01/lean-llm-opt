**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 &&\forall i\in N &&\text{(outgoing)}\\
&\sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 &&\forall j\in N &&\text{(incoming)}\\
&u_i-u_j+15\,x_{ij}\le 14 &&\forall i\in M,\;j\in M,\;i\neq j &&\text{(MTZ)}\\
&u_1=1 &&\text{(reference node)}\\
&x_{ij}\in\{0,1\} &&\forall i\neq j\\
&u_i\in\{1,\dots,15\} &&\forall i\in M
\end{aligned}
\]

where  

* \(N=\{1,2,\dots,15\}\) – all locations (location 1 is the depot).  
* \(M=N\setminus\{1\}\) – the 14 customer nodes.  
* \(d_{ij}\) – symmetric travel distance between locations \(i\) and \(j\).  
* \(x_{ij}\) – binary variable equal to 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\) – auxiliary position variable used in the MTZ subtour‑elimination constraints.

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` | For each row \(r\) (0‑based) and column header \(c\) (string of an integer) <br>• Node \(i = r+1\). <br>• Node \(j = \text{int}(c)\). <br>• If the cell contains a numeric value \(v\), set \(d_{ij}=v\). <br>• Set \(d_{ji}=d_{ij}\) (symmetry). <br>• For \(i=j\), set \(d_{ii}=0\). |
| \(N\) | Implicit | \(\{1,\dots,15\}\) |
| \(M\) | Implicit | \(N\setminus\{1\}\) |

The table `file_0_view_0` contains the upper‑triangular part of the distance matrix; the mapping rule above reconstructs the full symmetric matrix needed for the model. No other data are required.
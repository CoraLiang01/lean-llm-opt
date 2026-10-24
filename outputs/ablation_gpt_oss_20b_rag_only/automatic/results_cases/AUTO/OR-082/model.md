**Mathematical Model**

Let  

* \(N=\{Depot, A, B, C\}\) – set of all nodes.  
* \(C=\{A,B,C\}\) – set of customer nodes.  

Parameters  
* \(d_{ij}\) – road distance (km) from node \(i\) to node \(j\) (obtained from *DistanceMatrix.csv*).  

Decision variables  
* \(x_{ij}\in\{0,1\}\) – 1 if the van travels directly from \(i\) to \(j\), 0 otherwise, for all \(i,j\in N,\;i\neq j\).  
* \(u_i\in\{1,2,3\}\) – order in which customer \(i\) is visited, for all \(i\in C\).  

Objective  
\[
\min \;\; \sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Depot has one outgoing and one incoming arc**  
\[
\sum_{j\in C} x_{Depot,j}=1,\qquad 
\sum_{i\in C} x_{i,Depot}=1
\]

2. **Each customer has one outgoing and one incoming arc**  
\[
\sum_{j\in N,\;j\neq i} x_{i,j}=1,\qquad 
\sum_{j\in N,\;j\neq i} x_{j,i}=1,\qquad \forall i\in C
\]

3. **Sub‑tour elimination (MTZ)** – for all distinct customers \(i,j\in C\)  
\[
u_i - u_j + |C|\,x_{ij}\;\le\;|C|-1 \;\;\Longrightarrow\;\; u_i - u_j + 3\,x_{ij}\le 2
\]

4. **Order variable bounds**  
\[
1 \le u_i \le |C| \;\;\Longrightarrow\;\; 1 \le u_i \le 3,\qquad \forall i\in C
\]

5. **Binary arc variables**  
\[
x_{ij}\in\{0,1\},\qquad \forall i,j\in N,\;i\neq j
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | *DistanceMatrix.csv* (table_id: **file_0_view_0**) | For each pair \((i,j)\) with \(i\neq j\):<br>• Find the row where **Unnamed: 0** = \(i\).<br>• Take the value in the column named \(j\).<br>• Convert the string value to a numeric distance (km). |

*No other parameters are required; all other data are implicit in the model structure.*
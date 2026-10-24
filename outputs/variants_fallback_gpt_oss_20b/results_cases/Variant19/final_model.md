**Mathematical Model**

Let  

* \(N=\{\,\text{Node}\,\}\) be the set of all sites (from *network_nodes.csv*).  
* \(E=\{\,e=(i,j)\mid i,j\in N,\; \text{edge }(i,j)\text{ appears in } \text{network_edges.csv}\}\).  
* For each \(e=(i,j)\in E\) let \(c_e\) be its construction cost (column *ConstructionCost*).  
* \(r\in N\) be the root node (value of *RootNode* in *network_parameters.csv*).  
* \(M=|N|-1\) (the maximum possible flow on any arc).  

Define the directed arc set  
\[
A=\{\,a=(i,j)\mid (i,j)\in E\}\cup\{\,a=(j,i)\mid (i,j)\in E\}.
\]

Decision variables  

\[
\begin{aligned}
y_e &\in\{0,1\} &&\forall e\in E &&\text{(edge selected)}\\
f_a &\ge 0 &&\forall a\in A &&\text{(flow on directed arc)}
\end{aligned}
\]

Objective  

\[
\min \sum_{e\in E} c_e\,y_e
\]

Constraints  

1. **Tree size**  
\[
\sum_{e\in E} y_e = |N|-1
\]

2. **Root flow balance**  
\[
\sum_{a\in \delta^+(r)} f_a - \sum_{a\in \delta^-(r)} f_a = |N|-1
\]

3. **Non‑root flow balance**  
\[
\sum_{a\in \delta^+(i)} f_a - \sum_{a\in \delta^-(i)} f_a = -1
\qquad \forall i\in N\setminus\{r\}
\]

4. **Flow–edge linking**  
\[
f_{(i,j)} \le M\,y_{(i,j)} \qquad \forall (i,j)\in E
\]
(and similarly for the reverse arc \((j,i)\)).

5. **Variable domains**  
\[
y_e\in\{0,1\},\qquad f_a\ge 0
\]

---

**Data Mapping**

| Index / Parameter | Source Table | Source Columns | Mapping Rule |
|-------------------|--------------|----------------|--------------|
| \(N\) | *file_0_view_0* | `Node` | \(N=\{\text{Node}\}\) |
| \(E\) | *file_1_view_0* | `Node1`, `Node2` | \(E=\{(Node1,Node2)\}\) |
| \(c_e\) | *file_1_view_0* | `ConstructionCost` | \(c_{(i,j)}=\text{ConstructionCost}\) |
| \(r\) | *file_2_view_0* | `Parameter`=`RootNode`, `Value` | \(r=\text{Value}\) |
| \(M\) | – | – | \(M=|N|-1\) |
| \(A\) | – | – | \(A=\{(i,j),(j,i)\mid (i,j)\in E\}\) |
| Flow‑to‑edge linking | – | – | For each \(e=(i,j)\in E\), the two arcs \((i,j)\) and \((j,i)\) are linked to \(y_e\) via \(f_{(i,j)}\le M\,y_e\) and \(f_{(j,i)}\le M\,y_e\). |

All parameters are taken directly from the supplied CSV files; no additional data are introduced. The model is a standard integer linear program for a minimum‑cost spanning tree using a single‑commodity flow formulation.
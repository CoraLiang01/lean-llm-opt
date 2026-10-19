## Mathematical Model

**Sets**  
\(N\) – set of substations (nodes)  
\(E\) – set of undirected candidate links (edges)  
\(A=\{(i,j)\mid (i,j)\in E\text{ or }(j,i)\in E\}\) – set of directed arcs derived from \(E\)

**Parameters**  
\(c_e\) – construction cost of edge \(e\in E\)  
\(r\) – root node (the source of the flow)  

**Decision Variables**  
\(y_e\in\{0,1\}\)  for every \(e\in E\) (1 if link \(e\) is built)  
\(f_{ij}\ge 0\)  for every \((i,j)\in A\) (flow on directed arc \((i,j)\))

**Model**

\[
\begin{aligned}
\min\;&\sum_{e\in E} c_e\,y_e \\[4pt]
\text{s.t. } &\sum_{e\in E} y_e = |N|-1 &&\text{(exactly \(n-1\) links)}\\[4pt]
&f_{ij}\;\le\;( |N|-1)\,y_{e(i,j)} &&\forall (i,j)\in A &&\text{(flow only if link built)}\\[4pt]
&\sum_{(i,v)\in A} f_{i v}-\sum_{(v,j)\in A} f_{v j}=1 &&\forall v\in N\setminus\{r\} &&\text{(unit flow to each non‑root node)}\\[4pt]
&\sum_{(r,j)\in A} f_{r j}-\sum_{(i,r)\in A} f_{i r}=-(|N|-1) &&\text{(root sends one unit to each other node)}\\[4pt]
&y_e\in\{0,1\}\;\;\forall e\in E,\qquad
f_{ij}\ge 0\;\;\forall (i,j)\in A
\end{aligned}
\]

---

## Data Mapping Rules

| Parameter / Variable | Source Table | Source Columns | Mapping Rule |
|----------------------|--------------|----------------|--------------|
| \(N\) | `file_0_view_0` | `Node` | Each distinct value of `Node` forms an element of \(N\). |
| \(E\) | `file_1_view_0` | `Node1`, `Node2` | Each record defines an undirected edge \(e=\{Node1,Node2\}\). |
| \(c_e\) | `file_1_view_0` | `ConstructionCost` | For edge \(e=\{i,j\}\), \(c_e\) is the value in `ConstructionCost`. |
| \(r\) | `file_2_view_0` | `Value` where `Parameter`=`RootNode` | The single value in `Value` is the root node \(r\). |
| \(y_e\) | – | – | Binary decision variable associated with each edge \(e\in E\). |
| \(f_{ij}\) | – | – | Continuous non‑negative variable for each directed arc \((i,j)\in A\). |

*No numerical values are enumerated in the model; all coefficients are linked to the corresponding source columns as shown above.*
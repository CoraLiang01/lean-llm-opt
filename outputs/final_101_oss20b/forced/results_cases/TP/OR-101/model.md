**Mathematical model**

Let  

* \(I=\{A,B,\dots ,J\}\) (10 devices)  
* \(J=\{P1,P2,\dots ,P111\}\) (111 products)

Decision variables  
\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\;j\in J
\]
(continuous production quantity of product \(j\) on device \(i\))  

Parameters  

* \(p_j\) unit profit of product \(j\) \((j\in J)\)  
* \(t_{ij}\) processing time of product \(j\) on device \(i\) \((i\in I,\;j\in J)\)  
* \(c_i\) monthly operating capacity of device \(i\) \((i\in I)\)

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J} p_j\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{j\in J} t_{ij}\,x_{ij}\;\le\;c_i
      &&\forall i\in I \\[4pt]
&x_{ij}\;\ge\;0 &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source column |
|-----------|--------------|---------------|
| \(p_j\) | `file_2_view_0` | `Unit_Profit` |
| \(t_{ij}\) | `file_0_view_0` | `P1` … `P111` (columns) |
| \(c_i\) | `file_1_view_0` | `Monthly_Capacity` |

| Index set | Source table | Identifier column |
|-----------|--------------|-------------------|
| \(I\) | `file_0_view_0` (or `file_1_view_0`) | `Device` |
| \(J\) | `file_2_view_0` | `Product` |

These mappings use the exact identifiers and column names from the supplied CSV files; no aggregation or transformation is performed beyond the direct lookup.
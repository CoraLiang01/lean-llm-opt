**Mathematical model**

Let  

* \(I=\{A,B,C,D,E,F,G,H,I,J\}\) – set of devices  
* \(J=\{P1,P2,\dots ,P111\}\) – set of products  

Decision variable  

\[
x_j \;\ge 0 \qquad\forall j\in J
\]

(continuous production quantity of product \(j\))  

Parameters  

\[
\begin{aligned}
p_j &\;=\;\text{unit profit of product }j \\
t_{ij} &\;=\;\text{processing time of product }j\text{ on device }i \\
c_i &\;=\;\text{monthly operating capacity of device }i
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{j\in J} p_j\,x_j
\]

**Device‑capacity constraints**

\[
\sum_{j\in J} t_{ij}\,x_j \;\le\; c_i \qquad\forall i\in I
\]

**Non‑negativity**

\[
x_j \;\ge\; 0 \qquad\forall j\in J
\]

---

### Data‑mapping rules

| Parameter | Source table | Row axis | Column axis | Column name |
|-----------|--------------|----------|-------------|-------------|
| \(t_{ij}\) | `file_0_view_0` | `Device` | `Product` | – (matrix entry) |
| \(p_j\) | `file_2_view_0` | – | – | `Unit_Profit` |
| \(c_i\) | `file_1_view_0` | `Device` | – | `Monthly_Capacity` |

* `file_0_view_0` contains a 10 × 111 matrix of processing times; each row is a device, each column a product.  
* `file_2_view_0` lists the unit profit for each product.  
* `file_1_view_0` lists the monthly capacity for each device (missing entries are treated as unspecified in the data mapping).  

No other data are required. The model above is fully symbolic and ready for numerical solution.
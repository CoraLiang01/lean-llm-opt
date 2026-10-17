**Mathematical Model**

Let  

* \(I\) be the set of product types (indexed by *ProductName*).  
* \(J\) be the set of warehouses (indexed by *Warehouse ID*).  

Parameters  

\[
\begin{aligned}
v_i &\;=\; \text{Value of product } i \quad &(\text{table } \texttt{file\_1\_view\_0},\ \texttt{Value})\\
w_i &\;=\; \text{Weight of product } i \quad &(\text{table } \texttt{file\_1\_view\_0},\ \texttt{Weight})\\
C_j &\;=\; \text{Capacity of warehouse } j \quad &(\text{table } \texttt{file\_0\_view\_0},\ \texttt{Capacity})
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\; j\in J
\]

---

### Objective  

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\; v_i\,x_{ij}
\]

### Constraints  

1. **Warehouse capacity**  
   \[
   \sum_{i\in I} w_i\,x_{ij}\;\le\; C_j \qquad \forall\, j\in J
   \]

2. **Non‑negativity & integrality**  
   \[
   x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(v_i\)   | file_1_view_0 | Value |
| \(w_i\)   | file_1_view_0 | Weight |
| \(C_j\)   | file_0_view_0 | Capacity |

*Index sets*  
- \(I\) is derived from the distinct values in the **ProductName** column of **file_1_view_0**.  
- \(J\) is derived from the distinct values in the **Warehouse ID** column of **file_0_view_0**.
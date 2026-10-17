**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of shelves (indices given by *ShelfID*).  
* \(J=\{j\}\) be the set of products (indices given by *ProductName*).  

Parameters  

* \(c_i\) – weight capacity of shelf \(i\).  
* \(w_j\) – weight of one unit of product \(j\).  
* \(v_j\) – value of one unit of product \(j\).  

Decision variables  

* \(x_{ij}\ge 0\) – number of units of product \(j\) placed on shelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\ge\;0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Index |
|-----------|--------------|--------|-------|
| \(c_i\)   | `file_0_view_0` | `Capacity` | `ShelfID` |
| \(w_j\)   | `file_1_view_0` | `Weight` | `ProductName` |
| \(v_j\)   | `file_1_view_0` | `Value` | `ProductName` |

All indices \(i\) and \(j\) are taken from the complete set of rows returned in the observation. No additional constraints or bounds are imposed beyond those listed.
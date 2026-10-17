**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of display areas (DisplayID).  
- \(J=\{j\}\) be the set of vessel types (ProductName).  

Parameters  

- \(C_i\) – capacity of display area \(i\).  
- \(V_j\) – value of vessel type \(j\).  
- \(W_j\) – weight (size) of vessel type \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{+}\) – number of vessels of type \(j\) placed in display area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{+}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\)   | `file_0_view_0` | `Capacity` |
| \(V_j\)   | `file_1_view_0` | `Value` |
| \(W_j\)   | `file_1_view_0` | `Weight` |

Index sets \(I\) and \(J\) are taken from the distinct values of `DisplayID` and `ProductName` in the respective tables. No additional constraints or bounds are imposed beyond those listed.
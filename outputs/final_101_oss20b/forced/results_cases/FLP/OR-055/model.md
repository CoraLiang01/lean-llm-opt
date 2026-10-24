**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{DisplayID in } \text{file\_0\_view\_0}\,\}\)  
- \(J=\{\,j\mid j=\text{ProductName in } \text{file\_1\_view\_0}\,\}\)

Parameters  

- \(c_i\) – capacity of display area \(i\) (units of size)  
- \(v_j\) – value of one unit of product \(j\)  
- \(w_j\) – weight (size) of one unit of product \(j\)

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in display area \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are taken directly from the `DisplayID` and `ProductName` columns of the respective tables. No additional constraints or bounds are imposed beyond those stated.
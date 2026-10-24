**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a CabinetID in } \text{file\_0\_view\_0}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in } \text{file\_1\_view\_0}\}\)

Parameters  

- \(c_i\) – capacity of cabinet \(i\) (from column **Capacity** of *file_0_view_0*)  
- \(v_j\) – value per unit of product \(j\) (from column **Value** of *file_1_view_0*)  
- \(w_j\) – weight per unit of product \(j\) (from column **Weight** of *file_1_view_0*)

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in cabinet \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the unique values of `CabinetID` and `ProductName` in the respective tables. No additional constraints or bounds are imposed beyond those stated.
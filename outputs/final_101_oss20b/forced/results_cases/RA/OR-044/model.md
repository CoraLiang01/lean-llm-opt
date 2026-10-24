**Mathematical Model**

Let  

- \(I=\{\,i \mid i=\text{SectionID in } \text{file\_0\_view\_0}\,\}\)  
- \(J=\{\,j \mid j=\text{ProductName in } \text{file\_1\_view\_0}\,\}\)

Parameters  

- \(\text{Capacity}_i \; (i\in I)\) – available display space in section \(i\)  
- \(\text{Value}_j \; (j\in J)\) – revenue per unit of product \(j\)  
- \(\text{Weight}_j \; (j\in J)\) – space required per unit of product \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in section \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\;\sum_{j\in J}\; \text{Value}_j \; x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\; \text{Weight}_j \; x_{ij}\;\le\; \text{Capacity}_i,
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
\qquad \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` (indexed by `SectionID`) |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` (indexed by `ProductName`) |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` (indexed by `ProductName`) |

The index sets \(I\) and \(J\) are derived from the distinct values of `SectionID` and `ProductName` in the respective tables. No additional data or constraints are introduced beyond those specified.
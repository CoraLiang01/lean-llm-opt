**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{ShelfID}\,\}\) (sets of shelves)  
- \(J=\{\,j\mid j=\text{ProductName}\,\}\) (sets of products)  

Parameters  

- \(\text{Capacity}_i\) – shelf capacity (units of weight)  
- \(\text{Value}_j\) – monetary value per unit of product \(j\)  
- \(\text{Weight}_j\) – weight per unit of product \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\;\sum_{j\in J}\; \text{Value}_j \, x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\; \text{Weight}_j \, x_{ij}\;\le\; \text{Capacity}_i,
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{ShelfID}\) (index \(i\)) | `file_0_view_0` | `ShelfID` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |
| \(\text{ProductName}\) (index \(j\)) | `file_1_view_0` | `ProductName` |

All indices and parameters are bound directly to the corresponding rows and columns of the supplied CSV views. No additional data or constraints are introduced.
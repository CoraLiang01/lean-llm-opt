**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a ShelfID in } \texttt{capacity.csv}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in } \texttt{products.csv}\}\)

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\;=\; \text{Capacity of shelf } i \quad (i\in I)\\
\text{Value}_j &\;=\; \text{Value of product } j \quad (j\in J)\\
\text{Weight}_j &\;=\; \text{Weight of product } j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\ge 0 \quad\text{(units of product }j\text{ placed on shelf }i),\qquad i\in I,\;j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

Constraints  

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i,\qquad \forall\,i\in I
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the distinct values of `ShelfID` and `ProductName` in the respective tables. No additional constraints or integer requirements are imposed beyond non‑negativity of \(x_{ij}\).
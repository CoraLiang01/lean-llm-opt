**Symbolic Model**

Let  

\[
I=\{\,i \mid i \text{ is a ShelfID in } \texttt{capacity.csv}\,\},\qquad
J=\{\,j \mid j \text{ is a ProductName in } \texttt{products.csv}\,\}.
\]

Parameters  

\[
\begin{aligned}
\text{Cap}_i &\;=\; \text{Capacity of shelf } i, && i\in I,\\
\text{Val}_j &\;=\; \text{Value of product } j, && j\in J,\\
\text{Wgt}_j &\;=\; \text{Weight of product } j, && j\in J.
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad (i\in I,\;j\in J)
\]

( number of units of product \(j\) placed on shelf \(i\) ).

Let \(j^{*}\) denote the first product in the list of \(\texttt{products.csv}\) (the product with index 1 in the file, i.e. “Smartphone”).

**Objective**

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\;\text{Val}_j\,x_{ij}
\]

**Constraints**

1. **Weight capacity of each shelf**

\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad \forall\, i\in I
\]

2. **Minimum total quantity of the first product**

\[
\sum_{i\in I} x_{i\,j^{*}}\;\ge\;5
\]

3. **Non‑negativity and integrality**

\[
x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\;j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the `ShelfID` and `ProductName` columns of the respective tables. The first product \(j^{*}\) is the first record in `file_1_view_0` (row with `source_row` 0).
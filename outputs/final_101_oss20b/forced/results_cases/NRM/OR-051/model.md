**Mathematical Model**

Let  

\[
I=\{\,i\mid i\text{ is a cabinet}\,\},\qquad 
J=\{\,j\mid j\text{ is a coffee product}\,\}.
\]

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad\text{(weight capacity of cabinet }i),\\
\text{Value}_j     &\quad\text{(value per unit of product }j),\\
\text{Weight}_j    &\quad\text{(weight per unit of product }j).
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in\mathbb{Z}_{\ge 0}\quad\text{(units of product }j\text{ placed in cabinet }i).
\]

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i,
\qquad \forall\,i\in I,\\[4pt]
&x_{ij}\in\mathbb{Z}_{\ge 0},\qquad \forall\,i\in I,\;j\in J.
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) (Cabinet IDs) | `file_0_view_0` | `CabinetID` |
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(J\) (Product Names) | `file_1_view_0` | `ProductName` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

All indices and parameters are taken directly from the corresponding columns of the specified tables. No additional data or constraints are introduced beyond those required by the query.
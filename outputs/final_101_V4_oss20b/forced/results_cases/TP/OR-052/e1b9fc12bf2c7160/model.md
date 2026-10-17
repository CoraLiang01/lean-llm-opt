**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{BookshelfID from file\_0\_view\_0}\,\}\)  
- \(J=\{\,j\mid j=\text{ProductName from file\_1\_view\_0}\,\}\)

Parameters  

\[
\begin{aligned}
\text{capacity}[i] &\ge 0 &&\text{(weight capacity of bookshelf }i)\\
\text{value}[j] &\ge 0 &&\text{(value of book }j)\\
\text{weight}[j] &\ge 0 &&\text{(weight of book }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{+}\qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}[j]\;x_{ij}
\]

Constraints  

\[
\sum_{j\in J}\text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i]
\qquad \forall i\in I
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column |
|----------------------|--------------|--------|
| `capacity[i]` | `file_0_view_0` | `Capacity` |
| `value[j]` | `file_1_view_0` | `Value` |
| `weight[j]` | `file_1_view_0` | `Weight` |
| `x[i,j]` | – | – |

*Index sets*  
- `I` derived from all distinct `BookshelfID` values in `file_0_view_0`.  
- `J` derived from all distinct `ProductName` values in `file_1_view_0`.  

All parameters and variables are bound to the exact identifiers and values as supplied in the source tables. No additional data or constraints are introduced.
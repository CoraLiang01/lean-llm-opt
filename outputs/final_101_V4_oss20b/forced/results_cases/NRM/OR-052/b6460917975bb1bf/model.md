**Mathematical Model**

Let  

- \(I\) be the set of bookshelves (indexed by the values in *BookshelfID*).  
- \(J\) be the set of books (indexed by the values in *ProductName*).  

Parameters  

\[
\begin{aligned}
&\text{Capacity}_i &&\forall i\in I &&\text{(from column *Capacity* of table `file_0_view_0`)}\\
&\text{Value}_j &&\forall j\in J &&\text{(from column *Value* of table `file_1_view_0`)}\\
&\text{Weight}_j &&\forall j\in J &&\text{(from column *Weight* of table `file_1_view_0`)}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

Constraints  

1. **Capacity constraints (per bookshelf)**  

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall i\in I
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the distinct values of *BookshelfID* and *ProductName* in the respective tables. No additional data or constraints are introduced beyond those specified in the query.
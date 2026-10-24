**Mathematical Model**

Let  

- \(S\) be the set of shelves indexed by the values in column **ShelfID** of *file_0_view_0*.  
- \(P\) be the set of products indexed by the values in column **ProductName** of *file_1_view_0*.  

Parameters  

\[
\begin{aligned}
\text{Capacity}[s] &\quad \forall s\in S &&\text{(Shelf capacity)}\\
\text{Value}[p] &\quad \forall p\in P &&\text{(Product value)}\\
\text{Weight}[p] &\quad \forall p\in P &&\text{(Product weight)}
\end{aligned}
\]

Decision variables  

\[
x_{s p}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;p\in P
\]

Objective  

\[
\max \;\sum_{s\in S}\sum_{p\in P}\text{Value}[p]\;x_{s p}
\]

Capacity constraints  

\[
\sum_{p\in P}\text{Weight}[p]\;x_{s p}\;\le\;\text{Capacity}[s]
\qquad \forall s\in S
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| Capacity[s] | file_0_view_0 | Capacity |
| ShelfID (index set S) | file_0_view_0 | ShelfID |
| Value[p] | file_1_view_0 | Value |
| Weight[p] | file_1_view_0 | Weight |
| ProductName (index set P) | file_1_view_0 | ProductName |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.
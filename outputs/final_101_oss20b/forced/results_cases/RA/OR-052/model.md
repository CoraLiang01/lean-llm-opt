**Mathematical Model**

Let  

- \(I\) be the set of bookshelves indexed by **BookshelfID**.  
- \(J\) be the set of books indexed by **ProductName**.  

Parameters  

\[
\begin{aligned}
\text{Capacity}[i] &\;=\; \text{Capacity of bookshelf } i \quad (i\in I)\\
\text{Value}[j] &\;=\; \text{Value of book } j \quad (j\in J)\\
\text{Weight}[j] &\;=\; \text{Weight of book } j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\;=\;\;\text{number of units of book } j \text{ placed on bookshelf } i,
\qquad x_{ij}\in\mathbb{Z}_{\ge 0}\;\;(i\in I,\;j\in J)
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

**Constraints**

1. **Shelf capacity**

\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i],
\qquad \forall\, i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in\mathbb{Z}_{\ge 0},\qquad \forall\, i\in I,\;j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) (BookshelfID) | `file_0_view_0` | `BookshelfID` |
| \(\text{Capacity}[i]\) | `file_0_view_0` | `Capacity` |
| \(J\) (ProductName) | `file_1_view_0` | `ProductName` |
| \(\text{Value}[j]\) | `file_1_view_0` | `Value` |
| \(\text{Weight}[j]\) | `file_1_view_0` | `Weight` |

All indices, parameters, and variables are defined exactly as they appear in the supplied CSV data. No additional data or assumptions are introduced.
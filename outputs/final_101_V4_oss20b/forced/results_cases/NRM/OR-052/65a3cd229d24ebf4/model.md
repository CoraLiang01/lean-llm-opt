**Abstract Model**

**Index Sets**  
\(I\) – set of bookshelves (indexed by \(i\)).  
\(J\) – set of books (indexed by \(j\)).

**Parameters**  
\(\text{Capacity}_i\) – weight capacity of bookshelf \(i\).  
\(\text{Value}_j\) – monetary value of one unit of book \(j\).  
\(\text{Weight}_j\) – weight of one unit of book \(j\).

**Decision Variables**  
\(x_{ij}\) – number of units of book \(j\) placed on bookshelf \(i\).  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\).

**Objective Function**  
\[
\max \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

**Constraints**

1. **Capacity Constraints (per bookshelf)**  
\[
\sum_{j\in J} \text{Weight}_j \; x_{ij} \;\le\; \text{Capacity}_i , \qquad \forall i\in I
\]

2. **Non‑negativity and Integer Constraints**  
\[
x_{ij}\in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` (capacity.csv) | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` (products.csv) | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` (products.csv) | `Weight` |

*The table identifiers (`file_0_view_0`, `file_1_view_0`) and column names are taken directly from the CSVQA observation.*
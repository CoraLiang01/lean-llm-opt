**Symbolic Model**

Let  

- \(I\) be the set of shelves (indices from *capacity.csv*).  
- \(J\) be the set of products (indices from *products.csv*).  
- \(j^{*}\) denote the index of the first product (“Smartphone”).

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad (i\in I) &&\text{capacity of shelf }i,\\
\text{Value}_j &\quad (j\in J) &&\text{value of product }j,\\
\text{Weight}_j &\quad (j\in J) &&\text{weight of product }j.
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad (i\in I,\; j\in J)
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

Constraints  

1. **Shelf capacity (weight)**  

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall\, i\in I
\]

2. **Minimum quantity of the first product**  

\[
\sum_{i\in I}x_{ij^{*}}\;\ge\;5
\]

3. **Non‑negativity and integrality**  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` | For each row, `ShelfID` → index \(i\); value → \(\text{Capacity}_i\). |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` | For each row, `ProductName` → index \(j\); value → \(\text{Value}_j\). |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` | For each row, `ProductName` → index \(j\); value → \(\text{Weight}_j\). |
| \(j^{*}\) (first product) | `file_1_view_1` | `ProductName` | The single row where `ProductName` = “Smartphone” defines index \(j^{*}\). |

All other data (e.g., numeric values) are supplied by the referenced tables and are not enumerated in the symbolic model.
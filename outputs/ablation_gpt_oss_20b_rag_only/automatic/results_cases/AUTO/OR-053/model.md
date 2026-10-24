**Mathematical Model**

Let  

- \(S\) – set of shelves (indexed by `ShelfID` from *capacity.csv*)  
- \(P\) – set of products (indexed by `ProductName` from *products.csv*)

Parameters  

\[
\begin{aligned}
\text{Capacity}[s] &\;=\; \text{Capacity of shelf } s \quad (s\in S)\\
\text{Value}[p]     &\;=\; \text{value per unit of product } p \quad (p\in P)\\
\text{Weight}[p]    &\;=\; \text{weight per unit of product } p \quad (p\in P)
\end{aligned}
\]

Decision variables  

\[
x_{s,p}\;\in\;\mathbb{Z}_{\ge 0}\qquad (s\in S,\;p\in P)
\]

Objective  

\[
\max \;\sum_{s\in S}\;\sum_{p\in P}\;\text{Value}[p]\;x_{s,p}
\]

Capacity constraints (one per shelf)  

\[
\sum_{p\in P}\;\text{Weight}[p]\;x_{s,p}\;\le\;\text{Capacity}[s]
\qquad \forall\,s\in S
\]

All other variables are non‑negative integers.

---

**Data Mapping**

| Parameter / Variable | Source Table | Column(s) | Mapping Rule |
|----------------------|--------------|-----------|--------------|
| \(S\) | `file_0_view_0` | `ShelfID` | \(S=\{\,s \mid s=\text{ShelfID}\,\}\) |
| \(P\) | `file_1_view_0` | `ProductName` | \(P=\{\,p \mid p=\text{ProductName}\,\}\) |
| \(\text{Capacity}[s]\) | `file_0_view_0` | `Capacity` | \(\text{Capacity}[s]=\text{Capacity}\) where `ShelfID` = \(s\) |
| \(\text{Value}[p]\) | `file_1_view_0` | `Value` | \(\text{Value}[p]=\text{Value}\) where `ProductName` = \(p\) |
| \(\text{Weight}[p]\) | `file_1_view_0` | `Weight` | \(\text{Weight}[p]=\text{Weight}\) where `ProductName` = \(p\) |
| \(x_{s,p}\) | – | – | Decision variable, integer, \(x_{s,p}\ge 0\) |

All tables and columns are referenced exactly as returned by CSVQA. No additional data or constraints are introduced.
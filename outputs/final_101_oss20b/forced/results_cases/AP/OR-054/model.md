**Mathematical Model**

Let  

- \(I=\{i\mid\text{ShelfID}=i\}\) (sets of shelves)  
- \(J=\{j\mid\text{ProductName}=j\}\) (sets of products)

Parameters  

\[
\begin{aligned}
c_i &:= \text{Capacity of shelf } i && (i\in I)\\
v_j &:= \text{Value of product } j && (j\in J)\\
w_j &:= \text{Weight of product } j && (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{+}\qquad (i\in I,\;j\in J)
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

---

**Constraints**

1. Shelf capacity limits  

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall\, i\in I
\]

2. Non‑negativity (integrality)  

\[
x_{ij}\in \mathbb{Z}_{+}\qquad \forall\, i\in I,\;j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(c_i\)   | `file_0_view_0` | `ShelfID` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `ProductName` | `Value` |
| \(w_j\)   | `file_1_view_0` | `ProductName` | `Weight` |

All indices \(i\) and \(j\) are taken from the distinct values of the respective key columns in the source tables. No additional constraints or bounds are imposed beyond those listed.
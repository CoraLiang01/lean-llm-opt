**Mathematical Model**

Let  

\[
I=\{1,\dots ,10\}\quad\text{(shelves)},\qquad  
J=\{1,\dots ,20\}\quad\text{(products)}
\]

Parameters  

\[
\begin{aligned}
C_i &\;=\;\text{Capacity of shelf }i &&(i\in I)\\
w_j &\;=\;\text{Weight of one unit of product }j &&(j\in J)\\
v_j &\;=\;\text{Value of one unit of product }j &&(j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;=\;\text{number of units of product }j\text{ placed on shelf }i
\quad(i\in I,\;j\in J)
\]

\[
x_{ij}\in \mathbb{Z}_{\ge 0}
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**

Capacity of each shelf  

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; C_i \qquad \forall\, i\in I
\]

Non‑negativity (integrality already imposed)  

\[
x_{ij}\;\ge\;0 \qquad \forall\, i\in I,\;j\in J
\]

---

### Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C_i\) | Shelf capacity | `file_0_view_0` | `Capacity` |
| \(w_j\) | Product weight | `file_1_view_0` | `Weight` |
| \(v_j\) | Product value | `file_1_view_0` | `Value` |

All indices \(i\) and \(j\) are taken from the full sets of rows returned in the corresponding tables (`file_0_view_0` and `file_1_view_0`). No additional filtering is applied.
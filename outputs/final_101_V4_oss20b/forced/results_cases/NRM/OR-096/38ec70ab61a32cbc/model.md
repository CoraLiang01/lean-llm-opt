**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad
& Z = \sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr) \\[4pt]
\text{subject to}\quad
& \sum_{s\in S} x_{s,n}^{W}=P_{n}^{W}\qquad \forall n\in N \quad\text{(white population)}\\
& \sum_{s\in S} x_{s,n}^{NW}=P_{n}^{NW}\qquad \forall n\in N \quad\text{(non‑white population)}\\[4pt]
& \sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\le C_{s}\qquad \forall s\in S \quad\text{(school capacity)}\\[4pt]
& W_{s}\;\ge\;N_{s}\qquad \forall s\in S \quad\text{(white}\ge\text{non‑white)}\\
& 3\,W_{s}\;\le\;7\,N_{s}\qquad \forall s\in S \quad\text{(white}\le\frac{7}{3}\text{·non‑white)}\\[4pt]
& x_{s,n}^{W},\,x_{s,n}^{NW}\in\mathbb{Z}_{+}\qquad \forall s\in S,\;n\in N
\end{aligned}
\]

where  

\[
W_{s}=\sum_{n\in N}x_{s,n}^{W},\qquad 
N_{s}=\sum_{n\in N}x_{s,n}^{NW}.
\]

---

**Data Mapping**

| Symbol | Source Table | Source Column |
|--------|--------------|---------------|
| \(S\) | – | set of values in column **School** of *file_0_view_0* |
| \(N\) | – | set of values in column **Neighborhood** of *file_1_view_0* |
| \(C_{s}\) | *file_0_view_0* | **Capacity** |
| \(P_{n}^{W}\) | *file_1_view_0* | **Population_White** |
| \(P_{n}^{NW}\) | *file_1_view_0* | **Population_NonWhite** |
| \(d_{s,n}\) | *file_2_view_0* | column named after neighborhood \(n\) (e.g., **N01**, **N02**, …, **N31**) |

All parameters are taken directly from the corresponding CSV files; no numerical values are listed here. The model uses only the data specified above and respects the objective sense, variable domains, and constraints described in the query.
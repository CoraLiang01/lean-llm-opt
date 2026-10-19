## Cutting‑stock pattern‑selection model

**Index sets**  
\(I=\{A,B,C,D\}\) (item types)  
\(P=\{P1,P2,\dots ,P9\}\) (cutting patterns)

**Parameters**  
\(d_i\) demand of item \(i\) (from *item_demand.csv*)  
\(a_{i,p}\) units of item \(i\) produced by one roll cut with pattern \(p\) (from *cutting_patterns.csv*)

**Decision variables**  
\(y_p\in\mathbb{Z}_{+}\) number of rolls cut with pattern \(p\)

**Model**

\[
\begin{aligned}
\min\;&\sum_{p\in P} y_p \\[4pt]
\text{s.t.}\;&\sum_{p\in P} a_{i,p}\,y_p \;\ge\; d_i
\quad\forall\, i\in I \\[4pt]
&y_p \;\ge\; 0 \quad\forall\, p\in P \\[4pt]
&y_p \in \mathbb{Z}\quad\forall\, p\in P
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source | Mapping rule |
|-----------|--------|--------------|
| \(d_i\) | *file_0_view_0* (columns: Item, Demand) | \(d_i =\) value of column **Demand** where **Item** = \(i\) |
| \(a_{i,p}\) | *file_1_view_0* (columns: Pattern, A, B, C, D) | \(a_{i,p} =\) value of column corresponding to item \(i\) where **Pattern** = \(p\) |

*No other data are used.*
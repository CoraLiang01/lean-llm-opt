## Mathematical Model

**Index sets**  
\(I=\{A,B,C,D,E\}\) (items)  
\(P=\{P1,\dots ,P10\}\) (patterns)

**Parameters**  
\(d_i\) = demand of item \(i\) (from *item_demand.csv*, column *Demand*)  
\(a_{p,i}\) = number of pieces of item \(i\) produced by pattern \(p\) (from *cutting_patterns.csv*, columns *A*–*E*)

**Decision variables**  
\(y_p\in \mathbb{Z}_{\ge 0}\) = number of rolls cut with pattern \(p\)

**Objective**  
\[
\min \;\sum_{p\in P} y_p
\]

**Constraints**  
Demand satisfaction for each item  
\[
\sum_{p\in P} a_{p,i}\,y_p \;\ge\; d_i \qquad \forall i\in I
\]

Non‑negativity and integrality  
\[
y_p \;\ge\; 0,\; y_p\in \mathbb{Z}\qquad \forall p\in P
\]

---

## Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(d_i\)   | file_0_view_0 | Demand |
| \(a_{p,i}\) | file_1_view_0 | A, B, C, D, E (corresponding to item \(i\)) |

*Index sets \(I\) and \(P\) are derived from the distinct values in the respective columns of the source tables.*
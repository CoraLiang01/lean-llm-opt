**Mathematical Model**

Let  

- \(P\) be the set of platforms (indexed by \(i\)).  
- \(G\) be the set of games/genres (indexed by \(j\)).  

Parameters  

- \(c_i\) – memory capacity of platform \(i\).  
- \(v_j\) – predefined value of game \(j\).  
- \(w_j\) – memory weight of game \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of game \(j\) listed on platform \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in P}\sum_{j\in G} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in G} w_j\,x_{ij}\;\le\; c_i \quad \forall i\in P \\[4pt]
& x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in P,\; j\in G
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

*Index sets \(P\) and \(G\) are derived from the distinct values of `PlatformID` and `ProductName` in the respective tables.*
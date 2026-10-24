**Mathematical Model**

Let  

- \(P\) be the set of platforms (indexed by the values in column **PlatformID** of *capacity.csv*).  
- \(G\) be the set of games (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

- \(C_p\) – memory capacity of platform \(p\) (from *capacity.csv*).  
- \(V_g\) – predefined value of game \(g\) (from *products.csv*).  
- \(W_g\) – memory requirement of game \(g\) (from *products.csv*).  

Decision variables  

- \(x_{p g}\in \mathbb{Z}_{\ge 0}\) – number of units of game \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G} V_g\,x_{p g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G} W_g\,x_{p g}\;\le\; C_p , && \forall p\in P \\[4pt]
& x_{p g}\;\in\;\mathbb{Z}_{\ge 0}, && \forall p\in P,\; g\in G
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C_p\)   | `file_0_view_0` | `Capacity` |
| \(V_g\)   | `file_1_view_0` | `Value` |
| \(W_g\)   | `file_1_view_0` | `Weight` |

*Index sets*  
- \(P\) derived from all distinct values of `PlatformID` in `file_0_view_0`.  
- \(G\) derived from all distinct values of `ProductName` in `file_1_view_0`.
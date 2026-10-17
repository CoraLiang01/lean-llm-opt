**Mathematical Model**

Let  

- \(P\) be the set of platforms, indexed by the values of **PlatformId** in *capacity.csv*.  
- \(G\) be the set of game genres, indexed by the values of **ProductName** in *products.csv*.  

Parameters  

- \(C_p\) – memory capacity of platform \(p\).  
- \(V_g\) – predefined value of a unit of genre \(g\).  
- \(W_g\) – memory weight of a unit of genre \(g\).  

Decision variables  

- \(x_{p,g}\in \mathbb{Z}_{\ge 0}\) – number of units of genre \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G} V_g\,x_{p,g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G} W_g\,x_{p,g}\;\le\; C_p \qquad \forall p\in P \\[4pt]
& x_{p,g}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall p\in P,\;g\in G
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(C_p\)   | `file_0_view_0` | `PlatformId` | `Capacity` |
| \(V_g\)   | `file_1_view_0` | `ProductName` | `Value` |
| \(W_g\)   | `file_1_view_0` | `ProductName` | `Weight` |

The index sets \(P\) and \(G\) are derived from the distinct values of the key columns in the respective tables. No numerical values are enumerated in the model.
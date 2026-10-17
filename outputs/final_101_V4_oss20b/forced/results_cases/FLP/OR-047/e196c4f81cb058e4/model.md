**Mathematical Model**

Let  

- \(P\) be the set of platforms (identified by *PlatformId*).  
- \(G\) be the set of game genres (identified by *ProductName*).  

Parameters  

- \(\text{capacity}[p]\) – memory capacity of platform \(p\) (from *capacity.csv*).  
- \(\text{value}[g]\) – predefined value of a unit of genre \(g\) (from *products.csv*).  
- \(\text{weight}[g]\) – memory requirement of a unit of genre \(g\) (from *products.csv*).  

Decision variables  

- \(x_{p,g}\in \mathbb{Z}_{\ge 0}\) – number of units of genre \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G}\text{value}[g]\;x_{p,g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G}\text{weight}[g]\;x_{p,g}\;\le\;\text{capacity}[p],
\qquad \forall\,p\in P \\[4pt]
& x_{p,g}\in\mathbb{Z}_{\ge 0},\qquad \forall\,p\in P,\;g\in G .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(P\) | `file_0_view_0` | `PlatformId` | each distinct `PlatformId` |
| \(G\) | `file_1_view_0` | `ProductName` | each distinct `ProductName` |
| \(\text{capacity}[p]\) | `file_0_view_0` | `Capacity` | \(\text{capacity}[p] = \text{Capacity}\) for row with `PlatformId = p` |
| \(\text{value}[g]\) | `file_1_view_0` | `Value` | \(\text{value}[g] = \text{Value}\) for row with `ProductName = g` |
| \(\text{weight}[g]\) | `file_1_view_0` | `Weight` | \(\text{weight}[g] = \text{Weight}\) for row with `ProductName = g` |

No additional constraints or parameters are introduced beyond those explicitly required by the query.
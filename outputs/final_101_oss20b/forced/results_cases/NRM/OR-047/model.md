**Mathematical Model**

Let  

- \(P\) – set of platforms (indexed by the values in column **PlatformId** of *capacity.csv*).  
- \(G\) – set of game genres (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

- \(C_p\) – memory capacity of platform \(p\) (from **Capacity**).  
- \(v_g\) – value of one unit of genre \(g\) (from **Value**).  
- \(w_g\) – memory weight of one unit of genre \(g\) (from **Weight**).  

Decision variables  

- \(x_{p g}\) – number of units of genre \(g\) to list on platform \(p\).  
  \[
  x_{p g}\in \mathbb{Z}_{\ge 0}\quad \forall p\in P,\; g\in G
  \]

Objective  

\[
\max \;\sum_{p\in P}\sum_{g\in G} v_g\,x_{p g}
\]

Capacity constraints (one per platform)

\[
\sum_{g\in G} w_g\,x_{p g}\;\le\; C_p \qquad \forall p\in P
\]

All other constraints are the non‑negativity and integrality of \(x_{p g}\).

---

**Data Mapping**

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| \(C_p\) | `file_0_view_0` | `PlatformId`, `Capacity` |
| \(v_g\) | `file_1_view_0` | `ProductName`, `Value` |
| \(w_g\) | `file_1_view_0` | `ProductName`, `Weight` |

No additional data or constraints are introduced beyond those specified in the query.
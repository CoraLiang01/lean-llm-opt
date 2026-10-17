**Mathematical Model**

Let  

* \(P=\{1,\dots ,10\}\) – set of platforms (from *capacity.csv*).  
* \(G=\{1,\dots ,15\}\) – set of game genres (from *products.csv*).  

Parameters  

* \(c_p\) – memory capacity of platform \(p\) (units of memory).  
* \(v_g\) – value of one unit of genre \(g\).  
* \(w_g\) – memory weight of one unit of genre \(g\).  

Decision variables  

* \(x_{p,g}\in \mathbb{Z}_{\ge 0}\) – number of units of genre \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G} v_g\,x_{p,g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G} w_g\,x_{p,g}\;\le\; c_p \qquad \forall p\in P \\[4pt]
& x_{p,g}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall p\in P,\;g\in G
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(c_p\) | `file_0_view_0` | `Capacity` | One row per platform (PlatformId = \(p\)) |
| \(v_g\) | `file_1_view_0` | `Value` | One row per genre (ProductName = \(g\)) |
| \(w_g\) | `file_1_view_0` | `Weight` | One row per genre (ProductName = \(g\)) |

All indices \(p\) and \(g\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses only the parameters listed above and respects the integer domain for all decision variables.
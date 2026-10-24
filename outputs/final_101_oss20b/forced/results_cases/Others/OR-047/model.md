**Mathematical Model**

Let  

* \(P=\{1,\dots ,10\}\) – set of platforms (index from *PlatformId* in *capacity.csv*).  
* \(G=\{1,\dots ,15\}\) – set of game genres (index from *ProductName* in *products.csv*).  

Parameters  

* \(c_p\) – memory capacity of platform \(p\) (from *Capacity* column).  
* \(v_g\) – value of one unit of genre \(g\) (from *Value* column).  
* \(w_g\) – memory weight of one unit of genre \(g\) (from *Weight* column).  

Decision variables  

* \(x_{p g}\in \mathbb{Z}_{+}\) – number of units of genre \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G} v_g\,x_{p g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G} w_g\,x_{p g}\;\le\; c_p \qquad \forall p\in P \\[4pt]
& x_{p g}\;\in\;\mathbb{Z}_{+}\qquad \forall p\in P,\;g\in G
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(c_p\) | *capacity.csv* (table_id `file_0_view_0`) | `Capacity` | \(c_{\,\text{PlatformId}}\) |
| \(v_g\) | *products.csv* (table_id `file_1_view_0`) | `Value` | \(v_{\text{ProductName}}\) |
| \(w_g\) | *products.csv* (table_id `file_1_view_0`) | `Weight` | \(w_{\text{ProductName}}\) |

All indices \(p\) and \(g\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses only the parameters and constraints explicitly stated in the query.
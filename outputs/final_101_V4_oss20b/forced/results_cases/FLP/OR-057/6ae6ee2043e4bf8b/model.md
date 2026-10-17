**Mathematical Model**

Let  

* \(P=\{p\}\) – set of platforms (indexed by `PlatformID` in *capacity.csv*).  
* \(G=\{g\}\) – set of games (indexed by `ProductName` in *products.csv*).  

Parameters  

\[
\begin{aligned}
\text{capacity}_p &\quad\text{(memory capacity of platform }p)\\
\text{value}_g &\quad\text{(pre‑defined value of game }g)\\
\text{weight}_g &\quad\text{(memory requirement of game }g)
\end{aligned}
\]

Decision variables  

\[
x_{p g}\in \mathbb{Z}_{\ge 0}\qquad\text{(units of game }g\text{ listed on platform }p)
\]

---

**Objective**

\[
\max \;\sum_{p\in P}\sum_{g\in G}\text{value}_g\,x_{p g}
\]

---

**Constraints**

1. **Platform capacity**

\[
\sum_{g\in G}\text{weight}_g\,x_{p g}\;\le\;\text{capacity}_p,
\qquad \forall\,p\in P
\]

2. **Variable domain**

\[
x_{p g}\in\mathbb{Z}_{\ge 0}\qquad \forall\,p\in P,\;g\in G
\]

---

### Data Mapping

| Parameter | Source Table | Source Column | Index |
|-----------|--------------|---------------|-------|
| \(\text{capacity}_p\) | `file_0_view_0` | `Capacity` | `PlatformID` |
| \(\text{value}_g\) | `file_1_view_0` | `Value` | `ProductName` |
| \(\text{weight}_g\) | `file_1_view_0` | `Weight` | `ProductName` |

*Index sets*  
- \(P\) is the set of all `PlatformID` values in `file_0_view_0`.  
- \(G\) is the set of all `ProductName` values in `file_1_view_0`.

No additional constraints or parameters are required.
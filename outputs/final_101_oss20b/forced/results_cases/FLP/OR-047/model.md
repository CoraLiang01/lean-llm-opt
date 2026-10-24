**Mathematical Model**

Let  

- \(P\) be the set of platforms (indexed by the values of *PlatformId*).  
- \(G\) be the set of game genres (indexed by the values of *ProductName*).  

Parameters  

\[
\begin{aligned}
&\text{capacity}_{p} &&\text{(memory capacity of platform }p\text{)}\\
&\text{value}_{g} &&\text{(pre‑defined value of genre }g\text{)}\\
&\text{weight}_{g} &&\text{(memory requirement per unit of genre }g\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{p g}\in \mathbb{Z}_{\ge 0}\quad\text{(number of units of genre }g\text{ listed on platform }p)
\]

---

**Objective**

\[
\max \;\sum_{p\in P}\sum_{g\in G}\text{value}_{g}\;x_{p g}
\]

---

**Constraints**

1. **Platform capacity**

\[
\sum_{g\in G}\text{weight}_{g}\;x_{p g}\;\le\;\text{capacity}_{p},
\qquad \forall\,p\in P
\]

2. **Non‑negativity and integrality**

\[
x_{p g}\in \mathbb{Z}_{\ge 0},\qquad \forall\,p\in P,\;g\in G
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{capacity}_{p}\) | `file_0_view_0` | `Capacity` |
| \(\text{value}_{g}\) | `file_1_view_0` | `Value` |
| \(\text{weight}_{g}\) | `file_1_view_0` | `Weight` |

*Index sets*  
- \(P\) is the set of distinct values in the `PlatformId` column of `file_0_view_0`.  
- \(G\) is the set of distinct values in the `ProductName` column of `file_1_view_0`.  

No additional constraints or parameters are required.
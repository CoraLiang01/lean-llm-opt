**Mathematical Model**

Let  

- \(P\) be the set of platforms (indexed by `PlatformId`).  
- \(G\) be the set of game genres (indexed by `ProductName`).  

Parameters  

\[
\begin{aligned}
\text{Capacity}_p &\;=\; \text{Capacity of platform }p \quad (p\in P)\\
\text{Value}_g &\;=\; \text{Value of genre }g \quad (g\in G)\\
\text{Weight}_g &\;=\; \text{Memory requirement of genre }g \quad (g\in G)
\end{aligned}
\]

Decision variables  

\[
x_{p g}\;\in\;\mathbb{Z}_{\ge 0}\qquad (p\in P,\;g\in G)
\]

---

**Objective**

\[
\max \;\sum_{p\in P}\sum_{g\in G}\text{Value}_g\,x_{p g}
\]

---

**Constraints**

1. **Platform capacity**

\[
\sum_{g\in G}\text{Weight}_g\,x_{p g}\;\le\;\text{Capacity}_p
\qquad \forall\,p\in P
\]

2. **Non‑negativity & integrality**

\[
x_{p g}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\,p\in P,\;g\in G
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_p\) | `file_0_view_0` | `Capacity` (key: `PlatformId`) |
| \(\text{Value}_g\) | `file_1_view_0` | `Value` (key: `ProductName`) |
| \(\text{Weight}_g\) | `file_1_view_0` | `Weight` (key: `ProductName`) |

*Index sets*  
- \(P\) derived from all distinct `PlatformId` values in `file_0_view_0`.  
- \(G\) derived from all distinct `ProductName` values in `file_1_view_0`.
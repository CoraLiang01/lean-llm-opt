**Mathematical Model**

Let  

- \(P\) be the set of platforms indexed by `PlatformId`.  
- \(G\) be the set of game genres indexed by `ProductName`.  

Parameters  

\[
\begin{aligned}
\text{Capacity}_{p} &\quad p\in P &&\text{(memory capacity of platform }p)\\
\text{Value}_{g} &\quad g\in G &&\text{(value of genre }g)\\
\text{Weight}_{g} &\quad g\in G &&\text{(memory requirement of genre }g)
\end{aligned}
\]

Decision variables  

\[
x_{p g}\in \mathbb{Z}_{\ge 0}\qquad p\in P,\; g\in G
\]

---

**Objective**

\[
\max \;\sum_{p\in P}\sum_{g\in G}\text{Value}_{g}\;x_{p g}
\]

---

**Constraints**

1. **Platform capacity**

\[
\sum_{g\in G}\text{Weight}_{g}\;x_{p g}\;\le\;\text{Capacity}_{p},
\qquad \forall\,p\in P
\]

2. **Non‑negativity and integrality**

\[
x_{p g}\in \mathbb{Z}_{\ge 0},\qquad \forall\,p\in P,\; g\in G
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| `Capacity_p` | `file_0_view_0` | `Capacity` |
| `PlatformId` (index for \(P\)) | `file_0_view_0` | `PlatformId` |
| `Value_g` | `file_1_view_0` | `Value` |
| `Weight_g` | `file_1_view_0` | `Weight` |
| `ProductName` (index for \(G\)) | `file_1_view_0` | `ProductName` |

All indices and parameters are taken directly from the supplied tables; no additional data are introduced.
**Mathematical Model**

Let  

- \(P\) be the set of platforms indexed by the values in column **PlatformID** of *file_0_view_0*.  
- \(G\) be the set of game genres indexed by the values in column **ProductName** of *file_1_view_0*.  

Decision variables  

\[
x_{p,g}\in \mathbb{Z}_{\ge 0}\qquad \forall\,p\in P,\;g\in G
\]

Parameters  

\[
\begin{aligned}
\text{Capacity}_p &\;=\; \text{Capacity of platform }p \quad &&\text{from } \text{file\_0\_view\_0}.\text{Capacity} \\
\text{Value}_g     &\;=\; \text{value of genre }g \quad &&\text{from } \text{file\_1\_view\_0}.\text{Value} \\
\text{Weight}_g    &\;=\; \text{memory requirement of genre }g \quad &&\text{from } \text{file\_1\_view\_0}.\text{Weight}
\end{aligned}
\]

Objective  

\[
\max \;\sum_{p\in P}\;\sum_{g\in G}\;\text{Value}_g \; x_{p,g}
\]

Capacity constraints (one per platform)  

\[
\sum_{g\in G}\;\text{Weight}_g \; x_{p,g}\;\le\;\text{Capacity}_p
\qquad \forall\,p\in P
\]

All other constraints are implicit in the variable domain.

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| `Capacity_p` | `file_0_view_0` | `Capacity` |
| `Value_g` | `file_1_view_0` | `Value` |
| `Weight_g` | `file_1_view_0` | `Weight` |
| `PlatformID` (index for \(P\)) | `file_0_view_0` | `PlatformID` |
| `ProductName` (index for \(G\)) | `file_1_view_0` | `ProductName` |

All mappings use the exact identifiers and column names from the CSVQA observation. No numerical values are hard‑coded; the model is fully symbolic and ready for instantiation with the supplied data.
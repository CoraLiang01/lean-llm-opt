**Mathematical Model**

Let  

- \(P\) be the set of platforms (indexed by the values in *PlatformId*).  
- \(G\) be the set of game genres (indexed by the values in *ProductName*).  

Parameters (all taken directly from the source tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(C_p\) | Memory capacity of platform \(p\) | `file_0_view_0` – column **Capacity** |
| \(v_g\) | Value of a unit of genre \(g\) | `file_1_view_0` – column **Value** |
| \(w_g\) | Memory weight of a unit of genre \(g\) | `file_1_view_0` – column **Weight** |

Decision variables  

\[
x_{p g}\in \mathbb{Z}_{\ge 0}\qquad \forall p\in P,\; g\in G
\]

Objective  

\[
\max \;\sum_{p\in P}\sum_{g\in G} v_g\,x_{p g}
\]

Capacity constraints (one per platform)

\[
\sum_{g\in G} w_g\,x_{p g}\;\le\; C_p \qquad \forall p\in P
\]

Non‑negativity and integrality

\[
x_{p g}\;\ge\;0,\; x_{p g}\in \mathbb{Z}\qquad \forall p\in P,\; g\in G
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C_p\) | `file_0_view_0` | `Capacity` |
| \(v_g\) | `file_1_view_0` | `Value` |
| \(w_g\) | `file_1_view_0` | `Weight` |

All indices \(p\) and \(g\) are taken from the distinct values present in the corresponding columns of the source tables. No additional data or constraints are introduced beyond those specified in the query.
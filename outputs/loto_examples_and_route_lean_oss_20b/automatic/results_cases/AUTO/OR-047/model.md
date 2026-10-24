**Mathematical Model**

Let  

* \(P=\{1,\dots ,10\}\) – set of platforms (index \(p\)).  
* \(G=\{1,\dots ,15\}\) – set of game genres (index \(g\)).  

Parameters (all taken directly from the supplied tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(C_p\) | Memory capacity of platform \(p\) (in MB) | `capacity.csv`, column **Capacity** |
| \(v_g\) | Value of one unit of genre \(g\) (in arbitrary units) | `products.csv`, column **Value** |
| \(w_g\) | Memory weight of one unit of genre \(g\) (in MB) | `products.csv`, column **Weight** |

Decision variables  

\[
x_{p g}\in \mathbb{Z}_{\ge 0}\qquad \forall p\in P,\; g\in G
\]

---

### Objective
\[
\max \;\sum_{p\in P}\sum_{g\in G} v_g\,x_{p g}
\]

### Constraints
1. **Platform capacity**  
   \[
   \sum_{g\in G} w_g\,x_{p g}\;\le\; C_p \qquad \forall p\in P
   \]

2. **Non‑negativity & integrality**  
   \[
   x_{p g}\;\ge\;0,\; x_{p g}\in \mathbb{Z}\qquad \forall p\in P,\; g\in G
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C_p\) | `file_0_view_0` | `Capacity` |
| \(v_g\) | `file_1_view_0` | `Value` |
| \(w_g\) | `file_1_view_0` | `Weight` |

All indices \(p\) and \(g\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses only the data explicitly provided and respects the integer domain for all decision variables.
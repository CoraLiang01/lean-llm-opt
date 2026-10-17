**Mathematical Model**

Let  

- \(I\) be the set of storage areas.  
- \(J\) be the set of air‑conditioner product types.  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad (i\in I) \\
\text{Value}_j &\quad (j\in J) \\
\text{Weight}_j &\quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad (i\in I,\; j\in J)
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

---

**Constraints**

1. **Capacity limits**

\[
\sum_{j\in J} \text{Weight}_j \; x_{ij} \;\le\; \text{Capacity}_i
\qquad \forall\, i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) (StorageID) | `file_0_view_0` | `StorageID` |
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(J\) (ProductName) | `file_1_view_0` | `ProductName` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

All indices, parameters, and variables are defined exactly as above, with no additional data or constraints beyond those specified in the query.
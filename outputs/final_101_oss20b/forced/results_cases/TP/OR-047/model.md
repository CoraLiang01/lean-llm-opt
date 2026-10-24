**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of platforms (indices taken from *PlatformId*).  
* \(J=\{j\}\) be the set of game genres (indices taken from *ProductName*).  

Parameters  

* \(\text{cap}_i\) – memory capacity of platform \(i\) (from *Capacity*).  
* \(\text{val}_j\) – value of one unit of genre \(j\) (from *Value*).  
* \(\text{wt}_j\) – memory weight of one unit of genre \(j\) (from *Weight*).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{val}_j\,x_{ij}
\]

---

**Constraints**

1. **Capacity limits** – for each platform \(i\)

\[
\sum_{j\in J}\text{wt}_j\,x_{ij}\;\le\;\text{cap}_i
\]

2. **Non‑negativity & integrality** – already encoded in the variable definition.

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` | `cap[PlatformId] = Capacity` |
| \(\text{val}_j\) | `file_1_view_0` | `Value` | `val[ProductName] = Value` |
| \(\text{wt}_j\) | `file_1_view_0` | `Weight` | `wt[ProductName] = Weight` |

*Index sets*  
- \(I = \{ \text{PlatformId values from `file_0_view_0`}\}\)  
- \(J = \{ \text{ProductName values from `file_1_view_0`}\}\)

All parameters are taken directly from the supplied CSV files; no additional data or aggregation is introduced. The model is ready for use in any integer‑programming solver.
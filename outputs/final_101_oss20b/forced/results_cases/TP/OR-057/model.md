**Mathematical Model**

Let  

* \(I\) – set of platforms (indices obtained from *PlatformID* in *capacity.csv*).  
* \(J\) – set of games (indices obtained from *ProductName* in *products.csv*).  

Parameters  

\[
\begin{aligned}
\text{capacity}_i &\quad\text{(memory capacity of platform }i)\\
\text{value}_j &\quad\text{(value of game }j)\\
\text{weight}_j &\quad\text{(memory requirement of game }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad
\text{(number of units of game }j\text{ listed on platform }i)
\]

---

### Objective  

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{value}_j \; x_{ij}
\]

### Constraints  

1. **Capacity constraints (per platform)**  

\[
\sum_{j\in J} \text{weight}_j \; x_{ij}\;\le\; \text{capacity}_i
\qquad \forall i\in I
\]

2. **Non‑negativity & integrality**  

\[
x_{ij}\;\ge\;0,\qquad x_{ij}\in\mathbb{Z}
\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(I\) | `file_0_view_0` | `PlatformID` | Each distinct `PlatformID` becomes an index \(i\). |
| \(\text{capacity}_i\) | `file_0_view_0` | `Capacity` | `Capacity` value mapped to the corresponding \(i\). |
| \(J\) | `file_1_view_0` | `ProductName` | Each distinct `ProductName` becomes an index \(j\). |
| \(\text{value}_j\) | `file_1_view_0` | `Value` | `Value` value mapped to the corresponding \(j\). |
| \(\text{weight}_j\) | `file_1_view_0` | `Weight` | `Weight` value mapped to the corresponding \(j\). |

All indices, parameters, and variables are defined solely from the data in the two supplied CSV files; no additional data or assumptions are introduced.
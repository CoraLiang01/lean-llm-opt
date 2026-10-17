**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of platforms (PlatformID).  
* \(J=\{j\}\) be the set of games (ProductName).  

Parameters  

\[
\begin{aligned}
\text{Capacity}[i] &\quad\text{(memory capacity of platform }i)\\
\text{Value}[j] &\quad\text{(value of game }j)\\
\text{Weight}[j] &\quad\text{(memory requirement of game }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{Value}[j]\;x_{ij}
\]

---

**Constraints**

1. **Platform capacity**

\[
\sum_{j\in J} \text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]
\qquad \forall i\in I
\]

2. **Non‑negativity & integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column |
|-----------|----------|------------|--------------|
| `Capacity[i]` | `file_0_view_0` | `PlatformID` | `Capacity` |
| `Value[j]` | `file_1_view_0` | `ProductName` | `Value` |
| `Weight[j]` | `file_1_view_0` | `ProductName` | `Weight` |

*Index sets*  
- `I` is the set of all `PlatformID` values in `file_0_view_0`.  
- `J` is the set of all `ProductName` values in `file_1_view_0`.
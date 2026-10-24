**Mathematical Model**

Let  

- \(P=\{p\}\) be the set of platforms (indices from *PlatformID*).  
- \(J=\{j\}\) be the set of games (indices from *ProductName*).  

Parameters  

\[
\begin{aligned}
\text{Capacity}_p &\quad\text{(memory capacity of platform }p)\\
\text{Value}_j &\quad\text{(value of game }j)\\
\text{Weight}_j &\quad\text{(memory requirement of game }j)
\end{aligned}
\]

Decision variables  

\[
x_{p j}\in \mathbb{Z}_{\ge 0}\qquad \forall p\in P,\; j\in J
\]

Objective  

\[
\max \;\sum_{p\in P}\sum_{j\in J}\text{Value}_j\,x_{p j}
\]

Capacity constraints  

\[
\sum_{j\in J}\text{Weight}_j\,x_{p j}\;\le\;\text{Capacity}_p
\qquad \forall p\in P
\]

Non‑negativity (implicit in integer domain).  

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_p\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

Indices \(p\) and \(j\) are taken directly from the values in the `PlatformID` and `ProductName` columns of the respective tables.
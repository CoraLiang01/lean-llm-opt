**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) – suppliers  
* \(J=\{C1,\dots ,C12\}\) – customers  

Parameters  

\[
\begin{aligned}
s_i &\quad\text{(supply capacity of supplier }i)\\
d_j &\quad\text{(demand of customer }j)\\
c_{ij}&\quad\text{(per‑unit transportation cost from }i\text{ to }j)\\
f_{ij}&\quad\text{(fixed activation cost of route }i\!-\!j)\\
M_{ij}&\quad\text{(big‑M, e.g. }s_i\text{)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{(quantity shipped from }i\text{ to }j)\\
y_{ij}&\in\{0,1\} &&\text{(route }i\!-\!j\text{ used?)}
\end{aligned}
\]

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J}\bigl(c_{ij}\,x_{ij}+f_{ij}\,y_{ij}\bigr)\\[4pt]
\text{s.t. }&\sum_{i\in I}x_{ij}=d_j &&\forall j\in J\quad\text{(demand)}\\[4pt]
&\sum_{j\in J}x_{ij}\le s_i &&\forall i\in I\quad\text{(supply capacity)}\\[4pt]
&x_{ij}\le M_{ij}\,y_{ij} &&\forall i\in I,\;j\in J\quad\text{(link)}\\[4pt]
&x_{ij}\ge 0 &&\forall i,j\\
&y_{ij}\in\{0,1\} &&\forall i,j
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(s_i\) | `file_0_view_0` | `SupplyCapacity` | one row per supplier |
| \(d_j\) | `file_1_view_0` | `Demand` | one row per customer |
| \(c_{ij}\) | `file_2_view_0` | matrix columns `C1`–`C12` | row indexed by `Supplier` |
| \(f_{ij}\) | `file_3_view_0` | matrix columns `C1`–`C12` | row indexed by `Supplier` |
| \(M_{ij}\) | derived | \(s_i\) | big‑M chosen as supplier capacity |

All indices \(i\) and \(j\) are taken from the full sets of suppliers and customers returned by the respective tables; no additional filtering is applied. The model uses continuous shipment variables and binary route‑activation variables as specified.
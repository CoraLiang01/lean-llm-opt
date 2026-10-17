**Mathematical Model**

Let  

- \(P=\{S1,S2,S3,S4\}\) (plant indices)  
- \(C=\{C1,C2,C3,C4\}\) (customer indices)  

Decision variables  

\[
x_{p,c}\;\ge 0 \qquad \forall p\in P,\;c\in C
\]

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{demand of customer }c \\
s_p &\;=\;\text{production capacity of plant }p \\
c_{p,c} &\;=\;\text{transportation cost per unit from plant }p\text{ to customer }c
\end{aligned}
\]

Objective  

\[
\min \;\sum_{p\in P}\sum_{c\in C} c_{p,c}\,x_{p,c}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{p\in P} x_{p,c}=d_c \qquad \forall c\in C
\]

2. **Plant capacity**  
\[
\sum_{c\in C} x_{p,c}\le s_p \qquad \forall p\in P
\]

3. **Non‑negativity**  
\[
x_{p,c}\ge 0 \qquad \forall p\in P,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | `d_c = demand` where `customer = c` |
| \(s_p\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | `s_p = supply_capacity` where `Unnamed: 0 = p` |
| \(c_{p,c}\) | `file_2_view_0` | matrix `Unnamed: 0` (rows), `C1`–`C4` (columns) | `c_{p,c} = value` where row `Unnamed: 0 = p` and column header = `c` |

All indices \(p\) and \(c\) are taken from the full set of rows/columns present in the respective tables (no filtering).
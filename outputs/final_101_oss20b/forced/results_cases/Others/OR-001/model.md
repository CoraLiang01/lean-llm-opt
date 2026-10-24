**Mathematical Model**

Let  

- \(S=\{S1,\dots ,S18\}\) – distribution centers  
- \(C=\{C1,\dots ,C18\}\) – customer groups  

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{demand of customer }c \quad (c\in C)\\
s_s &\;=\;\text{daily supply capacity of center }s \quad (s\in S)\\
c_{sc}&\;=\;\text{transportation cost per unit from }s\text{ to }c \quad (s\in S,\;c\in C)
\end{aligned}
\]

Decision variables  

\[
x_{sc}\;\ge 0 \quad\text{(units shipped from }s\text{ to }c)
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{s\in S} x_{sc}=d_c \qquad \forall\,c\in C
\]

2. **Supply capacity**  
\[
\sum_{c\in C} x_{sc}\le s_s \qquad \forall\,s\in S
\]

3. **Non‑negativity**  
\[
x_{sc}\ge 0 \qquad \forall\,s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `demand` | `customer` → `c` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` | `Unnamed: 0` → `s` |
| \(c_{sc}\) | `file_2_view_0` | `C1`–`C18` | `Unnamed: 0` → `s`, column header → `c` |

All indices \(s\) and \(c\) are taken from the full set of rows/columns in the respective tables (no filtering). The model uses the exact numeric values as provided.
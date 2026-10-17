**Mathematical Model**

Let  

- \(I=\{S1,S2,\dots ,S10\}\) suppliers  
- \(J=\{C1,C2,\dots ,C10\}\) customers  

Parameters  

\[
\begin{aligned}
d_j &\quad\text{demand of customer }j\in J,\\
s_i &\quad\text{daily supply capacity of supplier }i\in I,\\
c_{ij}&\quad\text{transportation cost per unit from supplier }i\text{ to customer }j.
\end{aligned}
\]

Decision variables  

\[
x_{ij}\ge 0 \quad\text{units shipped from }i\text{ to }j,\qquad i\in I,\;j\in J.
\]

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}.
\]

Constraints  

1. **Customer demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J.
   \]

2. **Supplier capacity limits**  
   \[
   \sum_{j\in J} x_{ij}\le s_i,\qquad \forall i\in I.
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0,\qquad \forall i\in I,\;j\in J.
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(d_j\) | `file_0_view_0` | `demand` | For each record, `customer` → \(j\); `demand` → \(d_j\). |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | For each record, `Unnamed: 0` → \(i\); `supply_capacity` → \(s_i\). |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C10` | For each record, `Unnamed: 0` → \(i\); each column `Ck` → \(c_{ik}\) where \(j=Ck\). |

All indices \(i\) and \(j\) are taken from the union of identifiers present in the respective tables. No additional constraints or parameters are introduced beyond those explicitly required by the problem statement.
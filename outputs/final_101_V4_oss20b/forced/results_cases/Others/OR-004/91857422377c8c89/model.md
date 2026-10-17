**Mathematical Model**

Let  

* \(S=\{S1,\dots ,S12\}\) – set of distribution centers (suppliers).  
* \(C=\{C1,\dots ,C12\}\) – set of customer groups.  

Decision variables  

\[
x_{s,c}\;\ge 0 \qquad \forall s\in S,\;c\in C
\]

(continuous units shipped from supplier \(s\) to customer \(c\)).

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{demand of customer }c \\
u_s &\;=\;\text{supply capacity of supplier }s \\
c_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c
\end{aligned}
\]

**Objective**

\[
\min \; Z = \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\sum_{s\in S} x_{s,c} \;=\; d_c \qquad \forall c\in C
\]

2. **Supply capacity** – a supplier cannot ship more than its capacity  

\[
\sum_{c\in C} x_{s,c} \;\le\; u_s \qquad \forall s\in S
\]

3. **Non‑negativity** – already implied by variable definition  

\[
x_{s,c}\;\ge 0 \qquad \forall s\in S,\;c\in C
\]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | For each record, set \(d_{\text{customer}} = \text{demand}\). |
| \(u_s\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | For each record, set \(u_{\text{Unnamed: 0}} = \text{supply_capacity}\). |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier), columns `C1`–`C12` (customers) | For each record, for each customer column, set \(c_{\text{Unnamed: 0},\text{customer}} = \text{value}\). |

All index sets \(S\) and \(C\) are taken from the full set of identifiers present in the respective source tables; no additional filtering is applied. The model uses the exact numeric values as supplied; no rounding or aggregation is performed.
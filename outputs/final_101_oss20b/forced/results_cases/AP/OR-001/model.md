**Mathematical model**

Let  

* \(S\) – set of distribution centers (indices \(s\)).  
* \(C\) – set of customer groups (indices \(c\)).  

Parameters  

\[
\begin{aligned}
d_c &\quad\text{daily demand of customer }c,\\
s_s &\quad\text{daily supply capacity of center }s,\\
c_{sc} &\quad\text{transportation cost per unit from }s\text{ to }c .
\end{aligned}
\]

Decision variables  

\[
x_{sc}\;\ge 0 \quad\text{quantity shipped from }s\text{ to }c .
\]

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc} \\[4pt]
\text{s.t. } &\sum_{s\in S} x_{sc}= d_c \qquad \forall c\in C &&\text{(demand satisfied)}\\[4pt]
&\sum_{c\in C} x_{sc}\le s_s \qquad \forall s\in S &&\text{(supply limit)}\\[4pt]
&x_{sc}\ge 0 \qquad \forall s\in S,\;c\in C .
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Key column | Value column(s) | Mapping rule |
|-----------|--------------|------------|-----------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` | For each record, set \(d_{customer}=demand\). |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | For each record, set \(s_{Unnamed: 0}=supply_capacity\). |
| \(c_{sc}\) | `file_2_view_0` | `Unnamed: 0` | `C1` … `C18` | For each record, for each customer column `Ck`, set \(c_{Unnamed: 0,\;Ck}=value\). |

These rules bind the symbolic parameters to the exact CSV data without enumerating numerical values.
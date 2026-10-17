**Mathematical Model**

Let  

- \(S\) be the set of Walmart stores (indices \(s\)).  
- \(C\) be the set of customer groups (indices \(c\)).  

Parameters  

\[
\begin{aligned}
d_c &\quad\text{demand of customer }c,\\
s_s &\quad\text{daily supply capacity of store }s,\\
c_{sc} &\quad\text{transportation cost per unit from store }s\text{ to customer }c.
\end{aligned}
\]

Decision variables  

\[
x_{sc}\;\ge 0 \quad\text{units shipped from store }s\text{ to customer }c.
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{sc}=d_c \quad \forall\,c\in C
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C} x_{sc}\le s_s \quad \forall\,s\in S
   \]

3. **Non‑negativity**  
   \[
   x_{sc}\ge 0 \quad \forall\,s\in S,\;c\in C
   \]

---

**Data Mapping**

| Parameter | Source Table | ID Column | Value Column | Notes |
|-----------|--------------|-----------|--------------|-------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` | Customer demand |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Store supply capacity |
| \(c_{sc}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C12` (columns) | Cost matrix |

Index sets are derived from the unique values in the ID columns of the respective tables. No numerical values are enumerated in the model.
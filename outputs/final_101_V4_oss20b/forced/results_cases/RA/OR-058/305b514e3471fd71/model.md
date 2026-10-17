**Mathematical Model**

Let  

* \(S\) – set of suppliers (identified by the values in column **`Unnamed: 0`** of *fixed_cost.csv*).  
* \(C\) – set of customers (identified by the values in column **`customer`** of *demand.csv*).  

Parameters  

\[
\begin{aligned}
f_s &\;=\;\text{fixed cost of opening supplier }s 
      &&\text{(column `fixed_costs` of *fixed_cost.csv*)}\\
d_c &\;=\;\text{demand of customer }c 
      &&\text{(column `demand` of *demand.csv*)}\\
t_{s,c} &\;=\;\text{transportation cost per unit from supplier }s 
          \text{ to customer }c 
          &&\text{(columns `C1`–`C6` of *transportation_costs.csv*)}\\
M &\;=\;\sum_{c\in C} d_c 
      &&\text{(big‑\(M\) for linking constraints)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_s &\in \{0,1\} &&\text{(supplier }s\text{ is open)}\\
x_{s,c} &\in \mathbb{Z}_{\ge 0} &&\text{(units shipped from }s\text{ to }c)
\end{aligned}
\]

---

**Objective**

\[
\min \;\sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} t_{s,c}\,x_{s,c}
\]

---

**Constraints**

1. **Demand satisfaction** – each customer’s demand must be met

\[
\sum_{s\in S} x_{s,c} \;\ge\; d_c , \qquad \forall\,c\in C
\]

2. **Supplier activation linking** – shipments can occur only if the supplier is open

\[
x_{s,c} \;\le\; M\,y_s , \qquad \forall\,s\in S,\;\forall\,c\in C
\]

3. **Variable domains**

\[
y_s \in \{0,1\}, \quad x_{s,c} \in \mathbb{Z}_{\ge 0}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(f_s\) | `file_1_view_0` | `fixed_costs` (supplier id in `Unnamed: 0`) |
| \(d_c\) | `file_0_view_0` | `demand` (customer id in `customer`) |
| \(t_{s,c}\) | `file_2_view_0` | `C1`–`C6` (supplier id in `Unnamed: 0`) |
| \(M\) | – | computed as \(\sum_{c} d_c\) |

All indices \(s\) and \(c\) are taken directly from the corresponding identifier columns in the source tables. No numerical values are hard‑coded in the model; they are supplied at runtime via the data mapping above.
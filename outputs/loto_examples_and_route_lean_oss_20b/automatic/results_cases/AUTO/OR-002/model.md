## Mathematical Model

Let  

- \(S=\{S1,\dots ,S11\}\) be the set of Walmart stores (rows of *supply_capacity.csv*).  
- \(C=\{C1,\dots ,C12\}\) be the set of customer groups (rows of *customer_demand.csv*).  

Parameters  

- \(d_c\) – daily demand of customer \(c\) (from *customer_demand.csv*).  
- \(s_s\) – daily supply capacity of store \(s\) (from *supply_capacity.csv*).  
- \(c_{s,c}\) – transportation cost per unit from store \(s\) to customer \(c\) (from *transportation_costs.csv*).  

Decision variables  

- \(x_{s,c}\ge 0\) – quantity transported from store \(s\) to customer \(c\) (continuous).  

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\;&\sum_{s\in S} x_{s,c}= d_c \qquad &&\forall\,c\in C \quad\text{(demand satisfaction)}\\[4pt]
&\sum_{c\in C} x_{s,c}\le s_s \qquad &&\forall\,s\in S \quad\text{(store capacity)}\\[4pt]
&x_{s,c}\ge 0 \qquad &&\forall\,s\in S,\;c\in C
\end{aligned}
\]

---

## Data Mapping Rules

| Index Set | Source | Column(s) | Mapping |
|-----------|--------|-----------|---------|
| \(S\) | *supply_capacity.csv* (table_id `file_1_view_0`) | `Unnamed: 0` | Each distinct value defines an element of \(S\). |
| \(C\) | *customer_demand.csv* (table_id `file_0_view_0`) | `customer` | Each distinct value defines an element of \(C\). |

| Parameter | Source | Row Axis | Column Axis | Mapping |
|-----------|--------|----------|-------------|---------|
| \(d_c\) | *customer_demand.csv* (`file_0_view_0`) | `customer` | `demand` | For each row \(c\), \(d_c\) is the value in column `demand`. |
| \(s_s\) | *supply_capacity.csv* (`file_1_view_0`) | `Unnamed: 0` | `supply_capacity` | For each row \(s\), \(s_s\) is the value in column `supply_capacity`. |
| \(c_{s,c}\) | *transportation_costs.csv* (`file_2_view_0`) | `Unnamed: 0` (stores) | `C1`–`C12` (customers) | For each row \(s\) and column \(c\), \(c_{s,c}\) is the numeric value in that cell. |

All parameters are real numbers as provided in the CSV files. No additional constraints or transformations are applied beyond those explicitly stated in the model.
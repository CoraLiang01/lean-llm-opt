**Mathematical Model**

Let  

- \(S=\{\,\text{supplier1},\dots ,\text{supplier8}\,\}\) (Distribution centers)  
- \(C=\{\,\text{demand1},\dots ,\text{demand8}\,\}\) (Customer groups)  

Decision variables  

\[
x_{s,c}\;\ge 0 \qquad \forall s\in S,\;c\in C
\]

(Quantity shipped from supplier \(s\) to customer \(c\); continuous unless otherwise specified.)

Parameters  

\[
\begin{aligned}
\text{demand}_c &\;=\;\text{demand of customer }c\\
\text{supply}_s &\;=\;\text{available capacity of supplier }s\\
\text{cost}_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c
\end{aligned}
\]

Objective (minimise total transportation cost)

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}_{s,c}\;x_{s,c}
\]

Subject to  

1. **Demand fulfillment**  
   \[
   \sum_{s\in S}x_{s,c}= \text{demand}_c \qquad \forall c\in C
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C}x_{s,c}\;\le\;\text{supply}_s \qquad \forall s\in S
   \]

3. **Non‑negativity**  
   \[
   x_{s,c}\;\ge\;0 \qquad \forall s\in S,\;c\in C
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(\text{demand}_c\) | `file_0_view_0` | `Customers` | `demand` | For each row where `Customers = c`, assign `demand_c = demand`. |
| \(\text{supply}_s\) | `file_1_view_0` | `Supplier` | `supply_capacity` | For each row where `Supplier = s`, assign `supply_s = supply_capacity`. |
| \(\text{cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` | `demand1` … `demand8` | For each row where `Unnamed: 0 = s`, assign `cost_{s,c} = value in column c`. |

All indices \(S\) and \(C\) are taken directly from the rows of the respective tables; no additional sorting or synthesis of identifiers is performed.
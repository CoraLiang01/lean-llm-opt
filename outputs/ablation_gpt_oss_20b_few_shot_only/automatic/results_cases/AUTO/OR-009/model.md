**Mathematical Model**

Let  

\[
I=\{S1,S2,S3,S4\}, \qquad J=\{C1,C2,C3,C4\}
\]

\[
x_{ij}\;\ge 0 \quad\text{(units shipped from plant }i\text{ to outlet }j)
\]

\[
d_j \;\;(\text{daily demand of outlet }j), \qquad 
s_i \;\;(\text{production capacity of plant }i), \qquad 
c_{ij}\;\;(\text{transportation cost per unit from }i\text{ to }j)
\]

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(capacity)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source File | Key Column(s) | Mapping Rule |
|-----------|-------------|---------------|--------------|
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv` | `C1`, `C2`, `C3`, `C4` | For each customer \(j\), \(d_j\) is the value in the row whose identifier equals \(j\). |
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv` | `Unnamed: 0` (plant ID), `supply_capacity` | For each plant \(i\), \(s_i\) is the value in the row whose identifier equals \(i\). |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv` | `Unnamed: 0` (plant ID), columns `C1`–`C4` | For each pair \((i,j)\), \(c_{ij}\) is the value in the row whose identifier equals \(i\) and column equals \(j\). |

All identifiers (`S1`–`S4`, `C1`–`C4`) are taken verbatim from the source files. No additional data or aggregation is introduced.
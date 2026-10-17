**Mathematical Model**

Let  

* \(I=\{P_1,\dots ,P_{100}\}\) products  
* \(R=\{R_1,R_2,R_3\}\) resources  
* \(B=10\) units per batch  

Decision variable  
\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]
(number of batches of product \(i\) produced).

Parameters (all obtained from the CSV files)

\[
\begin{aligned}
&\text{profit}_i      &&\leftarrow \text{file\_0\_view\_0[product}=i].\text{profit\_per\_unit} \\
&r_{1,i}              &&\leftarrow \text{file\_0\_view\_0[product}=i].\text{r1\_per\_unit} \\
&r_{2,i}              &&\leftarrow \text{file\_0\_view\_0[product}=i].\text{r2\_per\_unit} \\
&r_{3,i}              &&\leftarrow \text{file\_0\_view\_0[product}=i].\text{r3\_per\_unit} \\
&\text{demand}_i      &&\leftarrow \text{file\_0\_view\_0[product}=i].\text{upper\_demand\_units} \\
&\text{cap}_r         &&\leftarrow \text{file\_1\_view\_0[resource}=r].\text{capacity}
\end{aligned}
\]

---

### Objective
\[
\max \;\; \sum_{i\in I} B\,x_i\,\text{profit}_i
\]

### Constraints

1. **Demand limits**  
   \[
   B\,x_i \;\le\; \text{demand}_i \qquad \forall i\in I
   \]

2. **Resource capacities**  
   \[
   \sum_{i\in I} B\,x_i\,r_{r,i} \;\le\; \text{cap}_r \qquad \forall r\in R
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Mapping Expression |
|-----------|--------------|--------|---------------------|
| \(I\) | file_0_view_0 | product | set of all distinct `product` values |
| \(R\) | file_1_view_0 | resource | set of all distinct `resource` values |
| \(\text{profit}_i\) | file_0_view_0 | profit_per_unit | `profit_per_unit` where `product = i` |
| \(r_{1,i}\) | file_0_view_0 | r1_per_unit | `r1_per_unit` where `product = i` |
| \(r_{2,i}\) | file_0_view_0 | r2_per_unit | `r2_per_unit` where `product = i` |
| \(r_{3,i}\) | file_0_view_0 | r3_per_unit | `r3_per_unit` where `product = i` |
| \(\text{demand}_i\) | file_0_view_0 | upper_demand_units | `upper_demand_units` where `product = i` |
| \(\text{cap}_r\) | file_1_view_0 | capacity | `capacity` where `resource = r` |
| \(B\) | file_0_view_0 | batch_size_units | constant value `10` (same for all products) |

All other model components are defined symbolically and do not require explicit numerical enumeration.
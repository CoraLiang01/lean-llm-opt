**Mathematical Model**

Let  

\[
P=\{F1,\dots ,F15\}, \qquad C=\{C1,\dots ,C15\}
\]

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_p &&\text{(opening cost of plant }p)\\
&\text{capacity}_p &&\text{(maximum units plant }p\text{ can supply)}\\
&\text{transport\_cost}_{p,c} &&\text{(per‑unit shipping cost from plant }p\text{ to customer }c)\\
&\text{demand}_c &&\text{(required units for customer }c)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_p &\in\{0,1\} &&\text{(plant }p\text{ is opened)}\\
&x_{p,c} &\ge 0 &&\text{(units shipped from }p\text{ to }c)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{p\in P}\text{fixed\_cost}_p\,y_p
      +\sum_{p\in P}\sum_{c\in C}\text{transport\_cost}_{p,c}\,x_{p,c}
\]

Constraints  

\[
\begin{aligned}
&\text{Demand satisfaction:}\qquad
   \sum_{p\in P}x_{p,c}= \text{demand}_c , &&\forall c\in C\\[4pt]
&\text{Capacity (only if opened):}\qquad
   \sum_{c\in C}x_{p,c}\le \text{capacity}_p\,y_p , &&\forall p\in P\\[4pt]
&\text{Variable domains:}\qquad
   x_{p,c}\ge 0,\; y_p\in\{0,1\}, &&\forall p\in P,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source file | Source column | Mapping rule |
|-----------|-------------|---------------|--------------|
| `fixed_cost_p` | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv` | `fixed_cost` | `fixed_cost_p = value in row where plant = p` |
| `capacity_p` | same file | `capacity` | `capacity_p = value in row where plant = p` |
| `transport_cost_{p,c}` | same file | `C1`–`C15` | `transport_cost_{p,c} = value in column corresponding to customer c in row where plant = p` |
| `demand_c` | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv` | `demand` | `demand_c = value in row where customer = c` |

Indices `p` and `c` are taken directly from the `plant` and `customer` identifiers in the respective CSV files. No additional limits or constants are introduced beyond those specified.
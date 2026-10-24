**Mathematical Model**

Let  

* \(I=\{C1,C2,\dots ,C111\}\) – set of component types  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – set of workshops  

Parameters  

\[
\begin{aligned}
t_{w,i} &\ge 0 &&\text{processing time per unit of component }i\text{ in workshop }w\\
p_i &\ge 0 &&\text{unit price of component }i\\
H_w &\ge 0 &&\text{total available working hours in workshop }w
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Objective**

\[
\max \; Z = \sum_{i\in I} p_i\,x_i
\]

---

**Constraints**

1. **Workshop capacity**

\[
\sum_{i\in I} t_{w,i}\,x_i \;\le\; H_w \qquad \forall w\in W
\]

2. **Non‑negativity & integrality**

\[
x_i \;\ge\; 0,\; x_i\in\mathbb{Z}\qquad \forall i\in I
\]

---

### Data Mapping

| Parameter | Source file | Key columns | Notes |
|-----------|-------------|-------------|-------|
| \(t_{w,i}\) | `Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv` | Row key: workshop name (`w`); Column key: component ID (`i`) | First column `Unnamed: 0` is ignored. |
| \(p_i\) | `Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv` | Row key: component ID (`i`); Column key: `unit_price` | First column `Unnamed: 0` is ignored. |
| \(H_w\) | `Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv` | Row key: workshop name (`w`); Column key: `total_hours` | |

The index sets \(I\) and \(W\) are taken directly from the row and column identifiers in the respective CSV files. No additional data are introduced beyond what is supplied.
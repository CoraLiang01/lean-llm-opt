**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(M=\{1,\dots ,101\}\) – set of radio models (model \(m\) is HiFi\(_m\))

Parameters (obtained from *workstation_times.csv*)

* \(t_{w,m}\) – processing time (minutes) per unit of model \(m\) at workstation \(w\)  
  \[
  t_{w,m}= \text{value of column ``HiFi}_{m}\text{_Minutes'' in row ``Workstation}=w
  \]
* \(p_w\) – maintenance percent at workstation \(w\)  
  \[
  p_w=\text{value of column ``Maintenance_Percent'' in row ``Workstation}=w
  \]
* Effective daily capacity of workstation \(w\)  
  \[
  C_w = 1440\,(1-\tfrac{p_w}{100})
  \]

Decision variables  

* \(x_m \in \mathbb{Z}_{\ge 0}\) – number of units of model \(m\) produced per day  
* (optional) \(I_w \ge 0\) – idle time at workstation \(w\)

---

\[
\begin{aligned}
\min\;&\sum_{w\in W} I_w \\[4pt]
\text{s.t.}\;& I_w = C_w - \sum_{m\in M} t_{w,m}\,x_m \qquad \forall w\in W \\[4pt]
& \sum_{m\in M} t_{w,m}\,x_m \;\le\; C_w \qquad \forall w\in W \\[4pt]
& x_m \in \mathbb{Z}_{\ge 0} \qquad \forall m\in M \\[4pt]
& I_w \ge 0 \qquad \forall w\in W
\end{aligned}
\]

*(The objective is equivalent to minimizing the total idle time, i.e. the sum of the effective capacities minus the total processing time used at each workstation.)*

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(t_{w,m}\) | *workstation_times.csv* (table `file_0_view_0`) | `t[w,m] = value of column "HiFi{m}_Minutes" in row where Workstation = w` |
| \(p_w\) | *workstation_times.csv* (table `file_0_view_0`) | `p[w] = value of column "Maintenance_Percent" in row where Workstation = w` |
| \(C_w\) | Computed | `C[w] = 1440 * (1 - p[w]/100)` |

*All indices \(w\) and \(m\) are taken from the rows and columns of the CSV file; no additional entities are introduced.*
**Mathematical Model**

Let  

* \(O\) be the set of all generation lot options (131 options).  
* For each option \(o\in O\)  
  * \(g_o\)  – generation per lot (kWh)  
  * \(c_o\)  – cost per lot (USD)  
  * \(t_o\)  – technology type (coal, gas, renewables)  

Decision variable  

\[
y_o \in \mathbb{Z}_{\ge 0}\quad\text{(number of lots of option }o\text{ purchased)}
\]

Objective (minimise total procurement cost)

\[
\min \;\sum_{o\in O} c_o\,y_o
\]

Demand‑satisfaction constraint (total generation must cover the 200‑kWh demand)

\[
\sum_{o\in O} g_o\,y_o \;\ge\; 200
\]

All other constraints are implicit in the variable domain (non‑negativity and integrality).

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(g_o\)   | `file_0_view_0` | `gen_per_lot` |
| \(c_o\)   | `file_0_view_0` | `cost_per_lot` |
| \(t_o\)   | `file_0_view_0` | `tech` |
| Index set \(O\) | `file_0_view_0` | `option` |

*The table `file_0_view_0` contains all 131 rows of `energy.csv`.  Each row defines one option \(o\) with its associated parameters.*
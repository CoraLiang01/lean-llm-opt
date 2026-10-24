**Mathematical Model**

Let  

* \(O\) be the set of all generation lot options (the *option* identifiers in *energy.csv*).  
* For each \(o\in O\)  
  * \(g_o\)  – generation per lot (from *gen_per_lot*).  
  * \(c_o\)  – cost per lot (from *cost_per_lot*).  

Decision variable  

\[
y_o \in \mathbb{Z}_{+}\quad\text{(number of lots of option }o\text{ purchased)}
\]

Objective  

\[
\min \;\sum_{o\in O} c_o\,y_o
\]

Demand‑satisfaction constraint  

\[
\sum_{o\in O} g_o\,y_o \;\ge\; 200
\]

(If the problem required exact fulfillment, replace “\(\ge\)” by “\(=\)”.)

All other constraints are the non‑negativity and integrality of \(y_o\).

---

**Data Mapping**

| Table ID | Column in CSV | Model Element | Notes |
|----------|---------------|---------------|-------|
| `file_0_view_0` | `option` | index \(o\) | Each distinct value becomes an element of \(O\). |
| `file_0_view_0` | `gen_per_lot` | \(g_o\) | Generation per lot for option \(o\). |
| `file_0_view_0` | `cost_per_lot` | \(c_o\) | Cost per lot for option \(o\). |
| `file_0_view_0` | `tech` | – | Not used in the model. |
| `file_0_view_0` | `unit_cost_est` | – | Not used in the model. |

The model uses all 131 rows of *energy.csv*; no aggregation or omission of data is performed.
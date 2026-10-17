**Mathematical Model**

Let  

* \(O\) be the set of all generation‑lot options returned in *energy.csv*  
  (131 options).  
* For each \(o\in O\)  
  * \(g_o\) – generation per lot (units) (from column **gen_per_lot**)  
  * \(c_o\) – cost per lot (yuan) (from column **cost_per_lot**)  

Decision variable  

* \(x_o \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(o\) to purchase.

Objective  

\[
\min \;\sum_{o\in O} c_o\,x_o
\]

Demand constraint  

\[
\sum_{o\in O} g_o\,x_o \;\ge\; 200
\]

No other constraints are required.

---

**Data Mapping**

| Parameter / Variable | Source | Table ID | Column | Notes |
|----------------------|--------|----------|--------|-------|
| \(O\) | All rows of *energy.csv* | `file_0_view_0` | `option` | Index set |
| \(g_o\) | `gen_per_lot` | `file_0_view_0` | `gen_per_lot` | Generation per lot |
| \(c_o\) | `cost_per_lot` | `file_0_view_0` | `cost_per_lot` | Cost per lot |
| \(x_o\) | Decision variable | – | – | Integer, non‑negative |

The `unit_cost_est` column is not used in the model. The demand of 200 units is a hard constraint applied to the sum of generation from all purchased lots. All variables and parameters are defined exactly as they appear in the source data.
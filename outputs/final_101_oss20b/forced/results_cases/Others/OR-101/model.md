## Mathematical Model

**Sets**  
\(P=\{P1,\dots ,P111\}\) products  
\(D=\{A,B,C,D,E,F,G,H,I,J\}\) devices  

**Parameters**  
\(p_i\) unit profit of product \(i\) (from *unit_product_profits.csv*)  
\(t_{d,i}\) processing time of product \(i\) on device \(d\) (from *device_time.csv*)  
\(c_d\) monthly capacity of device \(d\) (from *monthly_device_capacity.csv*)  

**Decision Variables**  
\(x_i\ge 0\) quantity of product \(i\) produced (continuous)  

**Objective**  
\[
\max \;\sum_{i\in P} p_i\,x_i
\]

**Constraints**  
For every device \(d\in D\):
\[
\sum_{i\in P} t_{d,i}\,x_i \;\le\; c_d
\]

---

## Data Mapping

| Table ID | Source File | Columns | Mapping |
|----------|-------------|---------|---------|
| `file_2_view_0` | `unit_product_profits.csv` | `Product`, `Unit_Profit` | \(p_{\text{Product}} = \text{Unit_Profit}\) |
| `file_0_view_0` | `device_time.csv` | `Device`, `P1`–`P111` | \(t_{\text{Device},\text{Product}} = \text{value}\) (missing entries treated as 0) |
| `file_1_view_0` | `monthly_device_capacity.csv` | `Device`, `Monthly_Capacity` | \(c_{\text{Device}} = \text{Monthly_Capacity}\) (missing entries treated as 0) |

*All indices \(i\) and \(d\) are taken from the full set of rows in the respective tables; no additional filtering is applied.*
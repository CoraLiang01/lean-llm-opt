**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{p\in P}\;\text{profit}_{p}\;x_{p} \\[4pt]
\text{s.t.}\;&\sum_{p\in P}\;\text{time}_{d,p}\;x_{p}\;\le\;\text{capacity}_{d},
\qquad \forall d\in D \\[4pt]
&x_{p}\;\ge\;0,\qquad \forall p\in P
\end{aligned}
\]

*Decision variables*  
\(x_{p}\) – continuous production quantity of product \(p\) (units).

*Sets*  
\(P\) – set of products \(\{P1,\dots ,P111\}\)  
\(D\) – set of devices \(\{A,B,C,D,E,F,G,H,I,J\}\)

*Parameters*  

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(\text{profit}_{p}\) | unit profit of product \(p\) | `file_2_view_0`, column `Unit_Profit`, key `Product` |
| \(\text{time}_{d,p}\) | processing time of product \(p\) on device \(d\) | `file_0_view_0`, key `Device`, columns `P1`–`P111` |
| \(\text{capacity}_{d}\) | monthly operating capacity of device \(d\) | `file_1_view_0`, column `Monthly_Capacity`, key `Device` |

**Data Mapping Rules**

```text
profit[p]          ←  table_id: file_2_view_0
                    key column: Product
                    value column: Unit_Profit

time[d,p]          ←  table_id: file_0_view_0
                    key column: Device
                    value columns: P1 … P111

capacity[d]        ←  table_id: file_1_view_0
                    key column: Device
                    value column: Monthly_Capacity
```

All parameters are taken directly from the corresponding CSV tables; no additional
values are invented.  The model uses continuous, non‑negative production
quantities and respects the device‑specific capacity limits.
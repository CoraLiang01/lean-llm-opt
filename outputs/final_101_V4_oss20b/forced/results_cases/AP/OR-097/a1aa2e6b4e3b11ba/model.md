**Mathematical Model**

Let  

* \(I=\{\,\text{Opt}_1,\dots ,\text{Opt}_{120}\,\}\) (120 options)  
* \(J=\{\,\text{Asset}_1,\dots ,\text{Asset}_6\,\}\) (6 underlying assets)

Parameters  

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(\text{Cost}[i]\) | per‑contract hedging cost of option \(i\) | `file_0_view_0` column **Cost** |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0` column **Delta** |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0` column **Gamma** |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0` column **Vega** |
| \(\text{MaxLong}[i]\) | maximum long position allowed for option \(i\) | `file_0_view_0` column **MaxLong** |
| \(\text{MaxShort}[i]\) | maximum short position allowed for option \(i\) (negative) | `file_0_view_0` column **MaxShort** |
| \(A[i,j]\) | binary indicator that option \(i\) references asset \(j\) | `file_1_view_0` columns **Asset_1**–**Asset_6** |
| \(\Delta_0\) | initial portfolio Delta | constant \(0.25\) |
| \(\Gamma_0\) | initial portfolio Gamma | constant \(0.08\) |
| \(\text{Vega}_0\) | initial portfolio Vega | constant \(0.17\) |
| \(T_\Delta\) | Delta tolerance band | constant \(0.06\) |
| \(T_\Gamma\) | Gamma tolerance band | constant \(0.05\) |
| \(T_{\text{Vega}}\) | Vega tolerance band | constant \(0.07\) |

Decision variables  

* \(x[i]\in \mathbb{Z}\) (number of contracts of option \(i\), positive = long, negative = short)

---

### Objective
\[
\min \;\sum_{i\in I} \text{Cost}[i]\;|x[i]|
\]

### Constraints

1. **Trading limits**  
   \[
   \text{MaxShort}[i] \;\le\; x[i] \;\le\; \text{MaxLong}[i] \qquad \forall i\in I
   \]

2. **Delta band**  
   \[
   -T_\Delta \;\le\; \Delta_0 + \sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x[i] \;\le\; T_\Delta
   \]

3. **Gamma band**  
   \[
   -T_\Gamma \;\le\; \Gamma_0 + \sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x[i] \;\le\; T_\Gamma
   \]

4. **Vega band**  
   \[
   -T_{\text{Vega}} \;\le\; \text{Vega}_0 + \sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x[i] \;\le\; T_{\text{Vega}}
   \]

---

**Data Mapping**

```json
{
  "I": {
    "source": "file_0_view_0",
    "identifier_column": "Option"
  },
  "J": {
    "identifiers": ["Asset_1","Asset_2","Asset_3","Asset_4","Asset_5","Asset_6"]
  },
  "Cost": {
    "table_id": "file_0_view_0",
    "column": "Cost"
  },
  "Delta": {
    "table_id": "file_0_view_0",
    "column": "Delta"
  },
  "Gamma": {
    "table_id": "file_0_view_0",
    "column": "Gamma"
  },
  "Vega": {
    "table_id": "file_0_view_0",
    "column": "Vega"
  },
  "MaxLong": {
    "table_id": "file_0_view_0",
    "column": "MaxLong"
  },
  "MaxShort": {
    "table_id": "file_0_view_0",
    "column": "MaxShort"
  },
  "A": {
    "table_id": "file_1_view_0",
    "columns": ["Asset_1","Asset_2","Asset_3","Asset_4","Asset_5","Asset_6"],
    "row_identifier": "Unnamed: 0"
  },
  "Delta_0": 0.25,
  "Gamma_0": 0.08,
  "Vega_0": 0.17,
  "T_Delta": 0.06,
  "T_Gamma": 0.05,
  "T_Vega": 0.07
}
```

This model captures the integer hedging decision, the cost minimization objective, the trading limits, and the risk‑band constraints using the full 120 × 6 asset‑reference matrix and option characteristics.
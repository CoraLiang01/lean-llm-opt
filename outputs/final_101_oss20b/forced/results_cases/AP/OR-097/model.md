**Mathematical Model**

Let  

* \(I=\{1,\dots ,120\}\) – set of option indices (option identifiers are the rows of the two CSV files).  
* \(J=\{1,\dots ,6\}\) – set of asset indices.  

Parameters (all taken directly from the CSV files)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Cost}[i]\) | per‑contract cost of option \(i\) | `file_0_view_0` column **Cost** |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0` column **Delta** |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0` column **Gamma** |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0` column **Vega** |
| \(\text{MaxLong}[i]\) | maximum long position for option \(i\) | `file_0_view_0` column **MaxLong** |
| \(\text{MaxShort}[i]\) | minimum short position for option \(i\) | `file_0_view_0` column **MaxShort** |
| \(A[i,j]\) | binary reference indicator (1 if option \(i\) references asset \(j\)) | `file_1_view_0` columns **Asset_1** … **Asset_6** |

Constants (given in the problem statement)

| Symbol | Value | Meaning |
|--------|-------|---------|
| \(\Delta_{\text{init}}\) | \(0.25\) | initial portfolio Delta |
| \(\Gamma_{\text{init}}\) | \(0.08\) | initial portfolio Gamma |
| \(\text{Vega}_{\text{init}}\) | \(0.17\) | initial portfolio Vega |
| \(\text{Tol}_{\Delta}\) | \(0.06\) | Delta tolerance |
| \(\text{Tol}_{\Gamma}\) | \(0.05\) | Gamma tolerance |
| \(\text{Tol}_{\text{Vega}}\) | \(0.07\) | Vega tolerance |

Decision variables

\[
x_i \in \mathbb{Z}\quad \forall i\in I
\]

---

### Objective

\[
\min \;\; \sum_{i\in I} \text{Cost}[i]\; |x_i|
\]

---

### Constraints

1. **Trading limits**

\[
\text{MaxShort}[i] \;\le\; x_i \;\le\; \text{MaxLong}[i] \qquad \forall i\in I
\]

2. **Risk‑band constraints** (one for each Greek)

\[
\bigl|\, \Delta_{\text{init}} + \sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x_i \,\bigr| \;\le\; \text{Tol}_{\Delta}
\]

\[
\bigl|\, \Gamma_{\text{init}} + \sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x_i \,\bigr| \;\le\; \text{Tol}_{\Gamma}
\]

\[
\bigl|\, \text{Vega}_{\text{init}} + \sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x_i \,\bigr| \;\le\; \text{Tol}_{\text{Vega}
}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(\text{Cost}[i]\) | `file_0_view_0` | `Cost` |
| \(\Delta[i]\) | `file_0_view_0` | `Delta` |
| \(\Gamma[i]\) | `file_0_view_0` | `Gamma` |
| \(\text{Vega}[i]\) | `file_0_view_0` | `Vega` |
| \(\text{MaxLong}[i]\) | `file_0_view_0` | `MaxLong` |
| \(\text{MaxShort}[i]\) | `file_0_view_0` | `MaxShort` |
| \(A[i,j]\) | `file_1_view_0` | `Asset_1` … `Asset_6` (columns 2–7) |

All indices \(i\) correspond to the `Option` column in `file_0_view_0` and the `Unnamed: 0` column in `file_1_view_0`.  
All indices \(j\) correspond to the asset columns `Asset_1` … `Asset_6`.  

The constants \(\Delta_{\text{init}}, \Gamma_{\text{init}}, \text{Vega}_{\text{init}}\) and the tolerances are fixed values supplied in the problem statement.
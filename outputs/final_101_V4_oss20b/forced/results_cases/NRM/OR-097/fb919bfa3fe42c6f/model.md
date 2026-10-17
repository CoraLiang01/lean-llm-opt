**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is an option in }{\tt OptionCharacteristics.csv}\}\)  
* \(J=\{1,2,3,4,5,6\}\)  (assets)

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Cost}[i]\) | per‑contract cost of option \(i\) | `file_0_view_0` – column **Cost** |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0` – column **Delta** |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0` – column **Gamma** |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0` – column **Vega** |
| \(\text{MaxLong}[i]\) | maximum long position for option \(i\) | `file_0_view_0` – column **MaxLong** |
| \(\text{MaxShort}[i]\) | minimum short position for option \(i\) | `file_0_view_0` – column **MaxShort** |
| \(A[i,j]\) | binary reference matrix (1 if option \(i\) references asset \(j\)) | `file_1_view_0` – columns **Asset_1** … **Asset_6** |
| \(\Delta_0,\Gamma_0,\text{Vega}_0\) | initial portfolio exposures | constants: 0.25, 0.08, 0.17 |
| \(\text{Tol}_\Delta,\text{Tol}_\Gamma,\text{Tol}_{\text{Vega}}\) | risk‑band tolerances | constants: 0.06, 0.05, 0.07 |

Decision variables  

* \(x[i]\in\mathbb Z\) – number of contracts of option \(i\) (positive = long, negative = short)

**Objective**

\[
\min \;\sum_{i\in I}\text{Cost}[i]\;|x[i]|
\]

**Constraints**

1. **Trading limits**

\[
\text{MaxShort}[i]\;\le\;x[i]\;\le\;\text{MaxLong}[i],\qquad \forall i\in I
\]

2. **Risk‑band constraints** (for each Greek \(G\in\{\Delta,\Gamma,\text{Vega}\}\))

\[
\Bigl|\,G_0 \;+\; \sum_{i\in I}\sum_{j\in J} G[i]\;A[i,j]\;x[i]\,\Bigr|
\;\le\;\text{Tol}_G
\]

where \(G[i]\) denotes \(\Delta[i],\Gamma[i]\) or \(\text{Vega}[i]\) respectively.

---

### Data Mapping

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Cost}[i]\) | `file_0_view_0` | **Cost** |
| \(\Delta[i]\) | `file_0_view_0` | **Delta** |
| \(\Gamma[i]\) | `file_0_view_0` | **Gamma** |
| \(\text{Vega}[i]\) | `file_0_view_0` | **Vega** |
| \(\text{MaxLong}[i]\) | `file_0_view_0` | **MaxLong** |
| \(\text{MaxShort}[i]\) | `file_0_view_0` | **MaxShort** |
| \(A[i,1]\) | `file_1_view_0` | **Asset_1** |
| \(A[i,2]\) | `file_1_view_0` | **Asset_2** |
| \(A[i,3]\) | `file_1_view_0` | **Asset_3** |
| \(A[i,4]\) | `file_1_view_0` | **Asset_4** |
| \(A[i,5]\) | `file_1_view_0` | **Asset_5** |
| \(A[i,6]\) | `file_1_view_0` | **Asset_6** |

All constants (\(\Delta_0,\Gamma_0,\text{Vega}_0,\text{Tol}_\Delta,\text{Tol}_\Gamma,\text{Tol}_{\text{Vega}}\)) are fixed values supplied in the query.
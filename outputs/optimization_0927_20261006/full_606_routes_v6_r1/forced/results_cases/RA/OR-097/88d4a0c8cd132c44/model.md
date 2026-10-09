Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A[i,j]$ be the binary indicator that option $i$ references asset $j$, $j=1,\dots,6$.

Let $\text{Cost}[i]$, $\text{Delta}[i]$, $\text{Gamma}[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ be as in OptionCharacteristics.csv, with option identifiers as below.

Constants:
- Initial net Delta: $0.25$
- Initial net Gamma: $0.08$
- Initial net Vega: $0.17$
- Tolerances: $|\Delta| \le 0.06$, $|\Gamma| \le 0.05$, $|\text{Vega}| \le 0.07$

---

**Objective:**
\[
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
\]

**Subject to:**

For all $i=1,\dots,120$ (option identifiers as below):

1. **Absolute value constraints:**
   \[
   y_i \ge x_i
   \]
   \[
   y_i \ge -x_i
   \]
   \[
   y_i \ge 0
   \]

2. **Trading limits:**
   \[
   \text{MaxShort}[i] \le x_i \le \text{MaxLong}[i]
   \]
   (with values as below)

3. **Delta risk constraint:**
   \[
   -0.06 \le 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Delta}[i] \cdot A[i,j] \cdot x_i \le 0.06
   \]

4. **Gamma risk constraint:**
   \[
   -0.05 \le 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Gamma}[i] \cdot A[i,j] \cdot x_i \le 0.05
   \]

5. **Vega risk constraint:**
   \[
   -0.07 \le 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \le 0.07
   \]

6. **Integrality:**
   \[
   x_i \in \mathbb{Z}, \quad \forall i=1,\dots,120
   \]

---

**Data (source order, OptionCharacteristics.csv):**

| Option   | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|----------|------|--------|-------|-------|---------|----------|
| Opt_1    | 9    | -0.54  | 0.12  | 0.10  | 9       | -14      |
| Opt_2    | 6    | 0.51   | 0.10  | 0.16  | 9       | -14      |
| Opt_3    | 13   | 0.17   | 0.02  | 0.19  | 10      | -7       |
| Opt_4    | 10   | -0.24  | 0.03  | 0.18  | 7       | -5       |
| Opt_5    | 7    | -0.61  | 0.14  | 0.11  | 12      | -9       |
| Opt_6    | 9    | -0.26  | 0.09  | 0.24  | 5       | -13      |
| Opt_7    | 12   | -0.24  | 0.01  | 0.20  | 10      | -5       |
| Opt_8    | 5    | 0.32   | 0.02  | 0.16  | 8       | -7       |
| Opt_9    | 9    | 0.19   | 0.10  | 0.17  | 5       | -8       |
| Opt_10   | 13   | 0.54   | 0.01  | 0.13  | 11      | -5       |
| ...      | ...  | ...    | ...   | ...   | ...     | ...      |
| Opt_120  | 12   | 0.54   | 0.03  | 0.19  | 9       | -7       |

(Full table continues for all 120 options, in the order and with the values as retrieved.)

**Data (source order, Option_AssetReferenceMatrix.csv):**

| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3    | 0       | 0       | 0       | 0       | 0       | 1       |
| Opt_4    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_5    | 0       | 0       | 1       | 0       | 0       | 0       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |
| Opt_120  | 0       | 1       | 0       | 0       | 1       | 0       |

(Full table continues for all 120 options, in the order and with the values as retrieved.)

---

**Summary of variables and indices:**
- $x_i$: integer, $i$ runs over all options in the order above (Opt_1, ..., Opt_120)
- $y_i$: continuous, $y_i \ge 0$, $y_i \ge x_i$, $y_i \ge -x_i$
- $A[i,j]$: as in Option_AssetReferenceMatrix.csv, $i=1,\dots,120$, $j=1,\dots,6$
- $\text{Cost}[i]$, $\text{Delta}[i]$, $\text{Gamma}[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$: as in OptionCharacteristics.csv, $i=1,\dots,120$

---

**All coefficients, bounds, and identifiers are as retrieved and must be used as above.**
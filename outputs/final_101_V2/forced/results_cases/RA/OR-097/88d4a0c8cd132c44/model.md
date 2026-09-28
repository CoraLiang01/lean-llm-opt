Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.
Let $y_i \geq 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $C_i$ = Cost of option $i$ (from OptionCharacteristics.csv, Option column Opt_$i$).
Let $D_i$ = Delta of option $i$.
Let $G_i$ = Gamma of option $i$.
Let $V_i$ = Vega of option $i$.
Let $A_{i,j}$ = 1 if option $i$ references asset $j$, 0 otherwise (from Option_AssetReferenceMatrix.csv, Option Opt_$i$, Asset_$j$).
Let $L_i$ = MaxLong for option $i$.
Let $S_i$ = MaxShort for option $i$.

Constants:
- Initial net Delta: $0.25$
- Initial net Gamma: $0.08$
- Initial net Vega: $0.17$
- Tolerances: $|\Delta| \leq 0.06$, $|\Gamma| \leq 0.05$, $|\text{Vega}| \leq 0.07$

Indices:
- $i = 1,\dots,120$ (options, OptionCharacteristics.csv, Option: Opt_$i$)
- $j = 1,\dots,6$ (assets, Option_AssetReferenceMatrix.csv, Asset_$j$)

Objective:
\[
\min \sum_{i=1}^{120} C_i \cdot y_i
\]

Subject to:

1. Absolute value constraints:
\[
y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0 \qquad \forall i=1,\dots,120
\]

2. Trading limits:
\[
S_i \leq x_i \leq L_i \qquad \forall i=1,\dots,120
\]

3. Delta risk constraint:
\[
-0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} D_i \cdot A_{i,j} \cdot x_i \leq 0.06
\]

4. Gamma risk constraint:
\[
-0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \leq 0.05
\]

5. Vega risk constraint:
\[
-0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} V_i \cdot A_{i,j} \cdot x_i \leq 0.07
\]

6. Integrality:
\[
x_i \in \mathbb{Z},\quad y_i \geq 0 \qquad \forall i=1,\dots,120
\]

All coefficients $C_i$, $D_i$, $G_i$, $V_i$, $L_i$, $S_i$ are taken directly from OptionCharacteristics.csv, preserving the Option identifier Opt_$i$ and source order. All $A_{i,j}$ are taken from Option_AssetReferenceMatrix.csv, using Option Opt_$i$ and Asset_$j$ columns, preserving source order.

This model uses all retrieved data, aligns all coefficients and identifiers as given, and respects all user-specified constraints and variable domains.
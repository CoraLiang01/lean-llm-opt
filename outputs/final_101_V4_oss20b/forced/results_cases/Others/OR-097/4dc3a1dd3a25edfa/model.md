## Mathematical Model

**Index sets**

\(I=\{\mathrm{Opt\_1},\ldots,\mathrm{Opt\_120}\}\) is the set of options and \(J=\{\mathrm{Asset\_1},\ldots,\mathrm{Asset\_6}\}\) is the set of assets.
Let \(\mathcal{G}=\{\Delta,\Gamma,\mathrm{Vega}\}\).

**Parameters**

For each option \(i\), let \(c_i\) be its per-contract cost, \(g_i^G\) its exposure for Greek \(G\), and \(L_i,U_i\) its short and long trading bounds. Let \(A_{ij}\in\{0,1\}\) indicate whether option \(i\) references asset \(j\).

\[
(b_\Delta,b_\Gamma,b_{\mathrm{Vega}})=(0.25,0.08,0.17),
\qquad
(\tau_\Delta,\tau_\Gamma,\tau_{\mathrm{Vega}})=(0.06,0.05,0.07).
\]

**Decision variables**

\[
x_i\in\mathbb{Z},\quad L_i\leq x_i\leq U_i,
\qquad y_i\in\mathbb{Z}_{\geq0}\qquad(i\in I).
\]

A positive position is long and a negative position is short. The auxiliary variable \(y_i\) represents \(|x_i|\) in the objective.

**Objective**

\[
\min\ \sum_{i\in I}c_i y_i.
\]

**Constraints**

Absolute-value linearization:

\[
y_i\geq x_i,\qquad y_i\geq-x_i\qquad(i\in I).
\]

Aggregate exposure bands:

\[
-\tau_G\leq b_G+\sum_{i\in I}\sum_{j\in J}g_i^G A_{ij}x_i\leq\tau_G
\qquad(G\in\mathcal{G}).
\]

All option costs are positive, so minimization sets \(y_i=|x_i|\) at optimality. An option referencing multiple assets contributes once per referenced asset, as required by the supplied asset-reference matrix.

**Data Mapping**

| Parameter | Table | Columns |
|---|---|---|
| Option IDs | `file_0_view_0` | `Option` |
| \(c_i\) | `file_0_view_0` | `Cost` |
| \(g_i^G\) | `file_0_view_0` | `Delta`, `Gamma`, `Vega` |
| \(L_i,U_i\) | `file_0_view_0` | `MaxShort`, `MaxLong` |
| \(A_{ij}\) | `file_1_view_0` | `Unnamed: 0`, `Asset_1` through `Asset_6` |

Options are created in characteristic-table order, followed by the auxiliary variables in the same order. Thus `C0` through `C119` represent positions and `C120` through `C239` represent absolute-position auxiliaries.

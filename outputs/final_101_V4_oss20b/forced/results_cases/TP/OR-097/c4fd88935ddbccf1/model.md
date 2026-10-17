**Mathematical model**

Let  

* \(I=\{1,\dots ,120\}\) – indices of the 120 options (option identifiers are the
  values of the column *Option* in *OptionCharacteristics.csv*).  
* \(J=\{1,\dots ,6\}\) – indices of the 6 underlying assets.

Parameters (all real numbers)

| Symbol | Definition | Source table & column |
|--------|------------|-----------------------|
| \(c_i\) | per‑contract hedging cost of option \(i\) | *file_0_view_0*, column **Cost** |
| \(\Delta_i\) | per‑contract Delta of option \(i\) | *file_0_view_0*, column **Delta** |
| \(\Gamma_i\) | per‑contract Gamma of option \(i\) | *file_0_view_0*, column **Gamma** |
| \(\nu_i\) | per‑contract Vega of option \(i\) | *file_0_view_0*, column **Vega** |
| \(L_i\) | maximum long position for option \(i\) | *file_0_view_0*, column **MaxLong** |
| \(S_i\) | maximum short position for option \(i\) | *file_0_view_0*, column **MaxShort** |
| \(a_{ij}\) | binary indicator that option \(i\) references asset \(j\) | *file_1_view_0*, columns **Asset_1**…**Asset_6** |
| \(\Delta^{\text{init}}\) | initial portfolio Delta | constant \(0.25\) |
| \(\Gamma^{\text{init}}\) | initial portfolio Gamma | constant \(0.08\) |
| \(\nu^{\text{init}}\) | initial portfolio Vega | constant \(0.17\) |
| \(\tau_\Delta\) | Delta tolerance band | constant \(0.06\) |
| \(\tau_\Gamma\) | Gamma tolerance band | constant \(0.05\) |
| \(\tau_\nu\) | Vega tolerance band | constant \(0.07\) |

Decision variables

| Symbol | Type | Bounds |
|--------|------|--------|
| \(x_i\) | integer | \(S_i \le x_i \le L_i\) |
| \(y_i\) | continuous, \(y_i \ge 0\) | \(y_i \ge |x_i|\) |

Auxiliary linearisation of the absolute value:

\[
\begin{aligned}
y_i &\ge x_i, \\
y_i &\ge -x_i, \qquad \forall i\in I .
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} c_i\, y_i .
\]

**Risk‑band constraints**

For each Greek \(G\in\{\Delta,\Gamma,\nu\}\) let \(G_i\) denote the corresponding
parameter (\(\Delta_i,\Gamma_i,\nu_i\)) and \(G^{\text{init}}\), \(\tau_G\) the
initial exposure and tolerance.  
The linearised absolute‑value constraints are

\[
\begin{aligned}
G^{\text{init}} + \sum_{i\in I}\sum_{j\in J} G_i\, a_{ij}\, x_i
&\le \tau_G ,\\[4pt]
-\Bigl(G^{\text{init}} + \sum_{i\in I}\sum_{j\in J} G_i\, a_{ij}\, x_i\Bigr)
&\le \tau_G ,
\end{aligned}
\qquad \forall G\in\{\Delta,\Gamma,\nu\}.
\]

**Complete model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} c_i\, y_i \\[4pt]
\text{s.t. } & y_i \ge x_i , \quad y_i \ge -x_i , \quad \forall i\in I \\[4pt]
& S_i \le x_i \le L_i , \quad \forall i\in I \\[4pt]
& \Delta^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \Delta_i\, a_{ij}\, x_i
   \le \tau_\Delta , \\[4pt]
& -\Bigl(\Delta^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \Delta_i\, a_{ij}\, x_i\Bigr)
   \le \tau_\Delta , \\[4pt]
& \Gamma^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \Gamma_i\, a_{ij}\, x_i
   \le \tau_\Gamma , \\[4pt]
& -\Bigl(\Gamma^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \Gamma_i\, a_{ij}\, x_i\Bigr)
   \le \tau_\Gamma , \\[4pt]
& \nu^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \nu_i\, a_{ij}\, x_i
   \le \tau_\nu , \\[4pt]
& -\Bigl(\nu^{\text{init}} + \sum_{i\in I}\sum_{j\in J} \nu_i\, a_{ij}\, x_i\Bigr)
   \le \tau_\nu , \\[4pt]
& x_i \in \mathbb{Z},\; y_i \ge 0 , \quad \forall i\in I .
\end{aligned}
\]

---

### Data‑mapping rules

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(c_i\) | *file_0_view_0* | **Cost** |
| \(\Delta_i\) | *file_0_view_0* | **Delta** |
| \(\Gamma_i\) | *file_0_view_0* | **Gamma** |
| \(\nu_i\) | *file_0_view_0* | **Vega** |
| \(L_i\) | *file_0_view_0* | **MaxLong** |
| \(S_i\) | *file_0_view_0* | **MaxShort** |
| \(a_{ij}\) | *file_1_view_0* | **Asset_1** … **Asset_6** (row identified by *Unnamed: 0*) |

All constants (\(\Delta^{\text{init}},\Gamma^{\text{init}},\nu^{\text{init}},\tau_\Delta,\tau_\Gamma,\tau_\nu\)) are hard‑coded as shown in the model.
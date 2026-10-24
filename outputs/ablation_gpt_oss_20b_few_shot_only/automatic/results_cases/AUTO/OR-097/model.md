**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{i\in I} \text{Cost}[i]\;|x_i| \\[4pt]
\text{subject to}\quad
& -T_{\Delta}\;\le\; \Delta_0 + \sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x_i \;\le\; T_{\Delta} \\[2pt]
& -T_{\Gamma}\;\le\; \Gamma_0 + \sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x_i \;\le\; T_{\Gamma} \\[2pt]
& -T_{\text{Vega}}\;\le\; \text{Vega}_0 + \sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x_i \;\le\; T_{\text{Vega}} \\[4pt]
& \text{MaxShort}[i] \;\le\; x_i \;\le\; \text{MaxLong}[i] \qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z} \qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\) | Set of option identifiers \(\{1,\dots,120\}\) | `OptionCharacteristics.csv` – column **Option** |
| \(J\) | Set of asset identifiers \(\{1,\dots,6\}\) | `Option_AssetReferenceMatrix.csv` – columns **Asset_1** … **Asset_6** |

**Parameters**

| Symbol | Meaning | Source File | Column |
|--------|---------|-------------|--------|
| \(\text{Cost}[i]\) | Per‑contract cost of option \(i\) | `OptionCharacteristics.csv` | `Cost` |
| \(\Delta[i]\) | Per‑contract Delta of option \(i\) | `OptionCharacteristics.csv` | `Delta` |
| \(\Gamma[i]\) | Per‑contract Gamma of option \(i\) | `OptionCharacteristics.csv` | `Gamma` |
| \(\text{Vega}[i]\) | Per‑contract Vega of option \(i\) | `OptionCharacteristics.csv` | `Vega` |
| \(\text{MaxLong}[i]\) | Maximum long position allowed for option \(i\) | `OptionCharacteristics.csv` | `MaxLong` |
| \(\text{MaxShort}[i]\) | Minimum short (negative) position allowed for option \(i\) | `OptionCharacteristics.csv` | `MaxShort` |
| \(A[i,j]\) | Binary indicator that option \(i\) references asset \(j\) | `Option_AssetReferenceMatrix.csv` | `Asset_1` … `Asset_6` |
| \(\Delta_0\) | Initial portfolio Delta | constant | 0.25 |
| \(\Gamma_0\) | Initial portfolio Gamma | constant | 0.08 |
| \(\text{Vega}_0\) | Initial portfolio Vega | constant | 0.17 |
| \(T_{\Delta}\) | Delta tolerance band | constant | 0.06 |
| \(T_{\Gamma}\) | Gamma tolerance band | constant | 0.05 |
| \(T_{\text{Vega}}\) | Vega tolerance band | constant | 0.07 |

**Decision Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_i\) | \(\mathbb{Z}\) | Integer number of contracts of option \(i\) (positive = long, negative = short) |

**Data Mapping**

```text
Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv
    Option          -> I
    Cost            -> Cost[i]
    Delta           -> Delta[i]
    Gamma           -> Gamma[i]
    Vega            -> Vega[i]
    MaxLong         -> MaxLong[i]
    MaxShort        -> MaxShort[i]

Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv
    Unnamed: 0      -> I
    Asset_1         -> A[i,1]
    Asset_2         -> A[i,2]
    Asset_3         -> A[i,3]
    Asset_4         -> A[i,4]
    Asset_5         -> A[i,5]
    Asset_6         -> A[i,6]
```

All constants (\(\Delta_0,\Gamma_0,\text{Vega}_0,T_{\Delta},T_{\Gamma},T_{\text{Vega}}\)) are fixed values supplied in the problem statement. The model above is a complete symbolic formulation ready for implementation in any mixed‑integer linear programming solver.
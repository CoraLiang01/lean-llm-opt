**Mathematical Model**

```
Sets
    I   : = {Option_1 , Option_2 , … , Option_120}          // options
    J   : = {Asset_1 , Asset_2 , Asset_3 , Asset_4 , Asset_5 , Asset_6}
    G   : = {Δ , Γ , ϑ}                                      // Greek symbols

Parameters
    Cost[i]          : cost per contract of option i
    Δ[i]             : delta per contract of option i
    Γ[i]             : gamma per contract of option i
    ϑ[i]             : vega per contract of option i
    MaxLong[i]       : maximum long position for option i
    MaxShort[i]      : maximum short position for option i
    A[i,j]           : 1 if option i references asset j, 0 otherwise

    Δ_initial = 0.25
    Γ_initial = 0.08
    ϑ_initial = 0.17

    Δ_tol = 0.06
    Γ_tol = 0.05
    ϑ_tol = 0.07

Variables
    x[i]   ∈ ℤ          // integer contracts (positive = long, negative = short)
    y[i]   ∈ ℤ, y[i] ≥ 0   // auxiliary variable for |x[i]|
    s[g]   ∈ ℝ          // net exposure for Greek g

Objective
    minimize  Σ_{i∈I}  Cost[i] · y[i]

Constraints
    // Absolute‑value linearisation for the objective
    y[i]  ≥  x[i]          ∀ i∈I
    y[i]  ≥ –x[i]          ∀ i∈I
    y[i]  ≥  0             ∀ i∈I

    // Trading limits
    MaxShort[i] ≤ x[i] ≤ MaxLong[i]   ∀ i∈I

    // Greek exposure constraints
    s[Δ] = Δ_initial + Σ_{i∈I} Σ_{j∈J} Δ[i] · A[i,j] · x[i]
    s[Γ] = Γ_initial + Σ_{i∈I} Σ_{j∈J} Γ[i] · A[i,j] · x[i]
    s[ϑ] = ϑ_initial + Σ_{i∈I} Σ_{j∈J} ϑ[i] · A[i,j] · x[i]

    –Δ_tol ≤ s[Δ] ≤ Δ_tol   ∀ g = Δ
    –Γ_tol ≤ s[Γ] ≤ Γ_tol   ∀ g = Γ
    –ϑ_tol ≤ s[ϑ] ≤ ϑ_tol   ∀ g = ϑ
```

---

**Data Mapping (CSVQA_DATA)**  

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| `Cost[i]` | `file_0_view_0` | `Cost` |
| `Δ[i]` | `file_0_view_0` | `Delta` |
| `Γ[i]` | `file_0_view_0` | `Gamma` |
| `ϑ[i]` | `file_0_view_0` | `Vega` |
| `MaxLong[i]` | `file_0_view_0` | `MaxLong` |
| `MaxShort[i]` | `file_0_view_0` | `MaxShort` |
| `A[i,j]` | `file_1_view_0` | `Asset_1` … `Asset_6` (columns) |
| `Δ_initial` | *constant* | 0.25 |
| `Γ_initial` | *constant* | 0.08 |
| `ϑ_initial` | *constant* | 0.17 |
| `Δ_tol` | *constant* | 0.06 |
| `Γ_tol` | *constant* | 0.05 |
| `ϑ_tol` | *constant* | 0.07 |

*Indices `i` and `j` are matched by the `Option` identifier in `file_0_view_0` and the `Unnamed: 0` column in `file_1_view_0`.  The Greek set `G` is fixed to the three symbols Δ, Γ, ϑ.*
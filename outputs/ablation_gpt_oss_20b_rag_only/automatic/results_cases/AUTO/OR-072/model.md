**Mathematical model**

```
Indices
    S = {1,2,…,24}          // hour‑shift indices

Parameters
    r_s   : required number of workers at shift s   (s ∈ S)

Variables
    x_s   ∈ ℤ_+   // number of workers who start at shift s   (s ∈ S)

Objective
    minimize   Z = Σ_{s∈S} x_s

Constraints
    ∀ t ∈ S :   Σ_{s∈S : s ≤ t ≤ s+3}  x_s   ≥   r_t
```

*Explanation*  
A worker who starts at shift s works continuously for the four consecutive shifts  
s, s+1, s+2, s+3.  The constraint for each shift t ensures that the total number of
workers covering that shift (those who started at s with s ≤ t ≤ s+3) is at least the
required number r_t.  The objective minimizes the total number of workers that
must be assigned.

---

**Data mapping**

| Parameter / Variable | Table ID          | Column(s) used | Key column(s) |
|----------------------|-------------------|----------------|---------------|
| r_s                  | `file_0_view_0`   | `Number Required` | `Shift`       |
| x_s                  | – (decision variable) | –              | –             |

*All rows of `file_0_view_0` are used; the `Shift` column provides the index `s` and the corresponding `Number Required` gives `r_s`.*
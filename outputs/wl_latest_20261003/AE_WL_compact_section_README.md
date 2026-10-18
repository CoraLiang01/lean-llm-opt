# Compact AE response section

`AE_WL_compact_section.tex` is a replacement section for the longer WL response.
It keeps the motivation, graph construction, cumulative WL formula, the primary
results for 101 original formulations and 36 variants, the category-specific
LOTO definition, and the interpretation needed for the AE's concern. The
alternative bounds-as-constraint analysis and repeated derivations are omitted.

The numerical source is the current reproducible package in
`outputs/wl_code_package_20261015/`. The LOTO values remove only the test
category's own reference pool, as in the ordinary leave-one-type-out rule.
The section also reports the separately saved performance evaluations (task
variants, pipeline LOTO, and forced routing) with their scope qualification.

The source assumes that the surrounding reply letter already loads `amsmath`,
`booktabs`, `float`, `natbib`, and `xcolor`, and defines the citation keys
`Gasse2019`, `Shervashidze2011`, and `Hosny2024`.

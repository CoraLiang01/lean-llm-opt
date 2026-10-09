\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 1091 assign[W11_P00] + 324 assign[W11_P01] + 379 assign[W11_P02] + 272 assign[W11_P03] + 1176 assign[W10_P00] + 1111 assign[W10_P01] + 1380 assign[W10_P02] + 542 assign[W10_P03] + 158 assign[W10_P04] + 922 assign[W10_P05] + 822 assign[W06_P01] + 223 assign[W06_P03] + 1155 assign[W06_P04] + 1055 assign[W06_P05] + 1397 assign[W02_P01] + 953 assign[W02_P02] + 714 assign[W02_P03] + 205 assign[W02_P04] + 1063 assign[W00_P00] + 219 assign[W00_P01] + 1329 assign[W00_P03] + 436 assign[W00_P05] + 642 assign[W04_P00] + 1130 assign[W04_P02] + 133 assign[W04_P03] + 199 assign[W04_P04] + 311 assign[W04_P05] + 102 assign[W12_P00] + 353 assign[W12_P01] + 651 assign[W12_P02] + 102 assign[W12_P03]
Subject To
  R0: + 1 assign[W11_P00] + 1 assign[W10_P00] + 1 assign[W00_P00] + 1 assign[W04_P00] + 1 assign[W12_P00] = 1
  R1: + 1 assign[W11_P01] + 1 assign[W10_P01] + 1 assign[W06_P01] + 1 assign[W02_P01] + 1 assign[W00_P01] + 1 assign[W12_P01] = 1
  R2: + 1 assign[W11_P02] + 1 assign[W10_P02] + 1 assign[W02_P02] + 1 assign[W04_P02] + 1 assign[W12_P02] = 1
  R3: + 1 assign[W11_P03] + 1 assign[W10_P03] + 1 assign[W06_P03] + 1 assign[W02_P03] + 1 assign[W00_P03] + 1 assign[W04_P03] + 1 assign[W12_P03] = 1
  R4: + 1 assign[W10_P04] + 1 assign[W06_P04] + 1 assign[W02_P04] + 1 assign[W04_P04] = 1
  R5: + 1 assign[W10_P05] + 1 assign[W06_P05] + 1 assign[W00_P05] + 1 assign[W04_P05] = 1
  R6: + 1 assign[W12_P00] + 1 assign[W12_P01] + 1 assign[W12_P02] + 1 assign[W12_P03] <= 1
  R7: + 1 assign[W06_P01] + 1 assign[W06_P03] + 1 assign[W06_P04] + 1 assign[W06_P05] <= 1
  R8: + 1 assign[W04_P00] + 1 assign[W04_P02] + 1 assign[W04_P03] + 1 assign[W04_P04] + 1 assign[W04_P05] <= 1
  R9: 0 assign[W11_P00] <= 1
  R10: + 1 assign[W11_P00] + 1 assign[W11_P01] + 1 assign[W11_P02] + 1 assign[W11_P03] <= 1
  R11: + 1 assign[W10_P00] + 1 assign[W10_P01] + 1 assign[W10_P02] + 1 assign[W10_P03] + 1 assign[W10_P04] + 1 assign[W10_P05] <= 1
  R12: + 1 assign[W02_P01] + 1 assign[W02_P02] + 1 assign[W02_P03] + 1 assign[W02_P04] <= 1
  R13: + 1 assign[W00_P00] + 1 assign[W00_P01] + 1 assign[W00_P03] + 1 assign[W00_P05] <= 1
Bounds
  0 <= assign[W11_P00] <= 1
  0 <= assign[W11_P01] <= 1
  0 <= assign[W11_P02] <= 1
  0 <= assign[W11_P03] <= 1
  0 <= assign[W10_P00] <= 1
  0 <= assign[W10_P01] <= 1
  0 <= assign[W10_P02] <= 1
  0 <= assign[W10_P03] <= 1
  0 <= assign[W10_P04] <= 1
  0 <= assign[W10_P05] <= 1
  0 <= assign[W06_P01] <= 1
  0 <= assign[W06_P03] <= 1
  0 <= assign[W06_P04] <= 1
  0 <= assign[W06_P05] <= 1
  0 <= assign[W02_P01] <= 1
  0 <= assign[W02_P02] <= 1
  0 <= assign[W02_P03] <= 1
  0 <= assign[W02_P04] <= 1
  0 <= assign[W00_P00] <= 1
  0 <= assign[W00_P01] <= 1
  0 <= assign[W00_P03] <= 1
  0 <= assign[W00_P05] <= 1
  0 <= assign[W04_P00] <= 1
  0 <= assign[W04_P02] <= 1
  0 <= assign[W04_P03] <= 1
  0 <= assign[W04_P04] <= 1
  0 <= assign[W04_P05] <= 1
  0 <= assign[W12_P00] <= 1
  0 <= assign[W12_P01] <= 1
  0 <= assign[W12_P02] <= 1
  0 <= assign[W12_P03] <= 1
Binaries
  assign[W11_P00] assign[W11_P01] assign[W11_P02] assign[W11_P03] assign[W10_P00] assign[W10_P01] assign[W10_P02] assign[W10_P03] assign[W10_P04] assign[W10_P05] assign[W06_P01] assign[W06_P03] assign[W06_P04] assign[W06_P05] assign[W02_P01] assign[W02_P02] assign[W02_P03] assign[W02_P04] assign[W00_P00] assign[W00_P01] assign[W00_P03] assign[W00_P05] assign[W04_P00] assign[W04_P02] assign[W04_P03] assign[W04_P04] assign[W04_P05] assign[W12_P00] assign[W12_P01] assign[W12_P02] assign[W12_P03]
End

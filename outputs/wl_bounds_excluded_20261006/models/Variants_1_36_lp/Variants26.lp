\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 147 assign[W10_P00] + 114 assign[W10_P01] + 998 assign[W10_P03] + 1270 assign[W10_P04] + 953 assign[W10_P07] + 109 assign[W19_P03] + 1306 assign[W19_P04] + 626 assign[W19_P05] + 466 assign[W19_P06] + 1365 assign[W19_P07] + 235 assign[W11_P00] + 1034 assign[W11_P02] + 782 assign[W11_P03] + 556 assign[W11_P04] + 1018 assign[W11_P05] + 1136 assign[W11_P07] + 430 assign[W00_P04] + 1385 assign[W00_P05] + 260 assign[W00_P06] + 1107 assign[W00_P07] + 675 assign[W20_P00] + 1055 assign[W20_P01] + 703 assign[W20_P03] + 893 assign[W20_P06] + 129 assign[W20_P07] + 543 assign[W04_P03] + 205 assign[W04_P04] + 1149 assign[W04_P05] + 533 assign[W04_P06] + 732 assign[W04_P07] + 1217 assign[W06_P00] + 425 assign[W06_P01] + 1320 assign[W06_P02] + 1097 assign[W06_P03] + 822 assign[W06_P04] + 221 assign[W06_P05] + 496 assign[W06_P06]
    + 908 assign[W06_P07] + 1361 assign[W14_P01] + 379 assign[W14_P03] + 239 assign[W14_P04] + 460 assign[W14_P05] + 130 assign[W14_P06] + 722 assign[W15_P00] + 577 assign[W15_P02] + 1062 assign[W15_P03] + 1314 assign[W15_P04] + 528 assign[W15_P05] + 835 assign[W15_P06] + 918 assign[W15_P07]
Subject To
  R0: + 1 assign[W10_P00] + 1 assign[W11_P00] + 1 assign[W20_P00] + 1 assign[W06_P00] + 1 assign[W15_P00] = 1
  R1: + 1 assign[W10_P01] + 1 assign[W20_P01] + 1 assign[W06_P01] + 1 assign[W14_P01] = 1
  R2: + 1 assign[W11_P02] + 1 assign[W06_P02] + 1 assign[W15_P02] = 1
  R3: + 1 assign[W10_P03] + 1 assign[W19_P03] + 1 assign[W11_P03] + 1 assign[W20_P03] + 1 assign[W04_P03] + 1 assign[W06_P03] + 1 assign[W14_P03] + 1 assign[W15_P03] = 1
  R4: + 1 assign[W10_P04] + 1 assign[W19_P04] + 1 assign[W11_P04] + 1 assign[W00_P04] + 1 assign[W04_P04] + 1 assign[W06_P04] + 1 assign[W14_P04] + 1 assign[W15_P04] = 1
  R5: + 1 assign[W19_P05] + 1 assign[W11_P05] + 1 assign[W00_P05] + 1 assign[W04_P05] + 1 assign[W06_P05] + 1 assign[W14_P05] + 1 assign[W15_P05] = 1
  R6: + 1 assign[W19_P06] + 1 assign[W00_P06] + 1 assign[W20_P06] + 1 assign[W04_P06] + 1 assign[W06_P06] + 1 assign[W14_P06] + 1 assign[W15_P06] = 1
  R7: + 1 assign[W10_P07] + 1 assign[W19_P07] + 1 assign[W11_P07] + 1 assign[W00_P07] + 1 assign[W20_P07] + 1 assign[W04_P07] + 1 assign[W06_P07] + 1 assign[W15_P07] = 1
  R8: 0 assign[W10_P00] <= 1
  R9: + 1 assign[W00_P04] + 1 assign[W00_P05] + 1 assign[W00_P06] + 1 assign[W00_P07] <= 1
  R10: + 1 assign[W19_P03] + 1 assign[W19_P04] + 1 assign[W19_P05] + 1 assign[W19_P06] + 1 assign[W19_P07] <= 1
  R11: + 1 assign[W11_P00] + 1 assign[W11_P02] + 1 assign[W11_P03] + 1 assign[W11_P04] + 1 assign[W11_P05] + 1 assign[W11_P07] <= 1
  R12: + 1 assign[W04_P03] + 1 assign[W04_P04] + 1 assign[W04_P05] + 1 assign[W04_P06] + 1 assign[W04_P07] <= 1
  R13: + 1 assign[W15_P00] + 1 assign[W15_P02] + 1 assign[W15_P03] + 1 assign[W15_P04] + 1 assign[W15_P05] + 1 assign[W15_P06] + 1 assign[W15_P07] <= 1
  R14: + 1 assign[W10_P00] + 1 assign[W10_P01] + 1 assign[W10_P03] + 1 assign[W10_P04] + 1 assign[W10_P07] <= 1
  R15: + 1 assign[W06_P00] + 1 assign[W06_P01] + 1 assign[W06_P02] + 1 assign[W06_P03] + 1 assign[W06_P04] + 1 assign[W06_P05] + 1 assign[W06_P06] + 1 assign[W06_P07] <= 1
  R16: + 1 assign[W20_P00] + 1 assign[W20_P01] + 1 assign[W20_P03] + 1 assign[W20_P06] + 1 assign[W20_P07] <= 1
  R17: + 1 assign[W14_P01] + 1 assign[W14_P03] + 1 assign[W14_P04] + 1 assign[W14_P05] + 1 assign[W14_P06] <= 1
Bounds
  0 <= assign[W10_P00] <= 1
  0 <= assign[W10_P01] <= 1
  0 <= assign[W10_P03] <= 1
  0 <= assign[W10_P04] <= 1
  0 <= assign[W10_P07] <= 1
  0 <= assign[W19_P03] <= 1
  0 <= assign[W19_P04] <= 1
  0 <= assign[W19_P05] <= 1
  0 <= assign[W19_P06] <= 1
  0 <= assign[W19_P07] <= 1
  0 <= assign[W11_P00] <= 1
  0 <= assign[W11_P02] <= 1
  0 <= assign[W11_P03] <= 1
  0 <= assign[W11_P04] <= 1
  0 <= assign[W11_P05] <= 1
  0 <= assign[W11_P07] <= 1
  0 <= assign[W00_P04] <= 1
  0 <= assign[W00_P05] <= 1
  0 <= assign[W00_P06] <= 1
  0 <= assign[W00_P07] <= 1
  0 <= assign[W20_P00] <= 1
  0 <= assign[W20_P01] <= 1
  0 <= assign[W20_P03] <= 1
  0 <= assign[W20_P06] <= 1
  0 <= assign[W20_P07] <= 1
  0 <= assign[W04_P03] <= 1
  0 <= assign[W04_P04] <= 1
  0 <= assign[W04_P05] <= 1
  0 <= assign[W04_P06] <= 1
  0 <= assign[W04_P07] <= 1
  0 <= assign[W06_P00] <= 1
  0 <= assign[W06_P01] <= 1
  0 <= assign[W06_P02] <= 1
  0 <= assign[W06_P03] <= 1
  0 <= assign[W06_P04] <= 1
  0 <= assign[W06_P05] <= 1
  0 <= assign[W06_P06] <= 1
  0 <= assign[W06_P07] <= 1
  0 <= assign[W14_P01] <= 1
  0 <= assign[W14_P03] <= 1
  0 <= assign[W14_P04] <= 1
  0 <= assign[W14_P05] <= 1
  0 <= assign[W14_P06] <= 1
  0 <= assign[W15_P00] <= 1
  0 <= assign[W15_P02] <= 1
  0 <= assign[W15_P03] <= 1
  0 <= assign[W15_P04] <= 1
  0 <= assign[W15_P05] <= 1
  0 <= assign[W15_P06] <= 1
  0 <= assign[W15_P07] <= 1
Binaries
  assign[W10_P00] assign[W10_P01] assign[W10_P03] assign[W10_P04] assign[W10_P07] assign[W19_P03] assign[W19_P04] assign[W19_P05] assign[W19_P06] assign[W19_P07] assign[W11_P00] assign[W11_P02] assign[W11_P03] assign[W11_P04] assign[W11_P05] assign[W11_P07] assign[W00_P04] assign[W00_P05] assign[W00_P06] assign[W00_P07] assign[W20_P00] assign[W20_P01] assign[W20_P03] assign[W20_P06] assign[W20_P07] assign[W04_P03] assign[W04_P04] assign[W04_P05] assign[W04_P06] assign[W04_P07] assign[W06_P00] assign[W06_P01] assign[W06_P02] assign[W06_P03] assign[W06_P04] assign[W06_P05] assign[W06_P06] assign[W06_P07] assign[W14_P01] assign[W14_P03] assign[W14_P04] assign[W14_P05] assign[W14_P06] assign[W15_P00] assign[W15_P02] assign[W15_P03] assign[W15_P04] assign[W15_P05] assign[W15_P06] assign[W15_P07]
End

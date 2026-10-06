\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  330 assign[W02_P00] + 193 assign[W02_P01] + 431 assign[W02_P04]
   + 1130 assign[W02_P06] + 644 assign[W02_P08] + 170 assign[W01_P01]
   + 919 assign[W01_P03] + 660 assign[W01_P04] + 1314 assign[W01_P05]
   + 668 assign[W01_P06] + 462 assign[W01_P08] + 614 assign[W01_P09]
   + 1079 assign[W10_P00] + 1128 assign[W10_P01] + 758 assign[W10_P02]
   + 108 assign[W10_P05] + 423 assign[W10_P07] + 744 assign[W10_P08]
   + 347 assign[W10_P09] + 538 assign[W17_P00] + 1359 assign[W17_P02]
   + 1366 assign[W17_P04] + 702 assign[W17_P05] + 122 assign[W17_P06]
   + 358 assign[W17_P08] + 238 assign[W18_P00] + 1085 assign[W18_P02]
   + 529 assign[W18_P03] + 582 assign[W18_P04] + 889 assign[W18_P05]
   + 139 assign[W18_P06] + 630 assign[W18_P08] + 538 assign[W18_P09]
   + 325 assign[W05_P00] + 772 assign[W05_P01] + 1042 assign[W05_P02]
   + 1394 assign[W05_P04] + 374 assign[W05_P06] + 1140 assign[W05_P08]
   + 127 assign[W05_P09] + 265 assign[W16_P00] + 1114 assign[W16_P01]
   + 1209 assign[W16_P03] + 231 assign[W16_P04] + 221 assign[W16_P05]
   + 135 assign[W16_P08] + 1116 assign[W00_P01] + 414 assign[W00_P02]
   + 554 assign[W00_P04] + 113 assign[W00_P05] + 853 assign[W00_P08]
   + 1142 assign[W00_P09] + 1143 assign[W11_P00] + 841 assign[W11_P01]
   + 920 assign[W11_P02] + 122 assign[W11_P03] + 634 assign[W11_P04]
   + 1250 assign[W11_P06] + 836 assign[W11_P07] + 180 assign[W11_P08]
   + 1105 assign[W11_P09] + 784 assign[W24_P00] + 1178 assign[W24_P01]
   + 1171 assign[W24_P02] + 103 assign[W24_P04] + 137 assign[W24_P05]
   + 832 assign[W24_P06] + 1279 assign[W24_P07] + 893 assign[W24_P08]
   + 351 assign[W24_P09] + 687 assign[W14_P01] + 126 assign[W14_P02]
   + 547 assign[W14_P03] + 125 assign[W14_P04] + 1358 assign[W14_P05]
   + 1391 assign[W14_P06] + 814 assign[W14_P07] + 1362 assign[W14_P08]
   + 962 assign[W14_P09]
Subject To
 R0: assign[W02_P00] + assign[W10_P00] + assign[W17_P00] + assign[W18_P00]
   + assign[W05_P00] + assign[W16_P00] + assign[W11_P00] + assign[W24_P00]
   = 1
 R1: assign[W02_P01] + assign[W01_P01] + assign[W10_P01] + assign[W05_P01]
   + assign[W16_P01] + assign[W00_P01] + assign[W11_P01] + assign[W24_P01]
   + assign[W14_P01] = 1
 R2: assign[W10_P02] + assign[W17_P02] + assign[W18_P02] + assign[W05_P02]
   + assign[W00_P02] + assign[W11_P02] + assign[W24_P02] + assign[W14_P02]
   = 1
 R3: assign[W01_P03] + assign[W18_P03] + assign[W16_P03] + assign[W11_P03]
   + assign[W14_P03] = 1
 R4: assign[W02_P04] + assign[W01_P04] + assign[W17_P04] + assign[W18_P04]
   + assign[W05_P04] + assign[W16_P04] + assign[W00_P04] + assign[W11_P04]
   + assign[W24_P04] + assign[W14_P04] = 1
 R5: assign[W01_P05] + assign[W10_P05] + assign[W17_P05] + assign[W18_P05]
   + assign[W16_P05] + assign[W00_P05] + assign[W24_P05] + assign[W14_P05]
   = 1
 R6: assign[W02_P06] + assign[W01_P06] + assign[W17_P06] + assign[W18_P06]
   + assign[W05_P06] + assign[W11_P06] + assign[W24_P06] + assign[W14_P06]
   = 1
 R7: assign[W10_P07] + assign[W11_P07] + assign[W24_P07] + assign[W14_P07]
   = 1
 R8: assign[W02_P08] + assign[W01_P08] + assign[W10_P08] + assign[W17_P08]
   + assign[W18_P08] + assign[W05_P08] + assign[W16_P08] + assign[W00_P08]
   + assign[W11_P08] + assign[W24_P08] + assign[W14_P08] = 1
 R9: assign[W01_P09] + assign[W10_P09] + assign[W18_P09] + assign[W05_P09]
   + assign[W00_P09] + assign[W11_P09] + assign[W24_P09] + assign[W14_P09]
   = 1
 R10: assign[W00_P01] + assign[W00_P02] + assign[W00_P04] + assign[W00_P05]
   + assign[W00_P08] + assign[W00_P09] <= 1
 R11: assign[W18_P00] + assign[W18_P02] + assign[W18_P03] + assign[W18_P04]
   + assign[W18_P05] + assign[W18_P06] + assign[W18_P08] + assign[W18_P09]
   <= 1
 R12: assign[W02_P00] + assign[W02_P01] + assign[W02_P04] + assign[W02_P06]
   + assign[W02_P08] <= 1
 R13: assign[W17_P00] + assign[W17_P02] + assign[W17_P04] + assign[W17_P05]
   + assign[W17_P06] + assign[W17_P08] <= 1
 R14: assign[W10_P00] + assign[W10_P01] + assign[W10_P02] + assign[W10_P05]
   + assign[W10_P07] + assign[W10_P08] + assign[W10_P09] <= 1
 R15: assign[W05_P00] + assign[W05_P01] + assign[W05_P02] + assign[W05_P04]
   + assign[W05_P06] + assign[W05_P08] + assign[W05_P09] <= 1
 R16: assign[W11_P00] + assign[W11_P01] + assign[W11_P02] + assign[W11_P03]
   + assign[W11_P04] + assign[W11_P06] + assign[W11_P07] + assign[W11_P08]
   + assign[W11_P09] <= 1
 R17: assign[W14_P01] + assign[W14_P02] + assign[W14_P03] + assign[W14_P04]
   + assign[W14_P05] + assign[W14_P06] + assign[W14_P07] + assign[W14_P08]
   + assign[W14_P09] <= 1
 R18: assign[W01_P01] + assign[W01_P03] + assign[W01_P04] + assign[W01_P05]
   + assign[W01_P06] + assign[W01_P08] + assign[W01_P09] <= 1
 R19: assign[W24_P00] + assign[W24_P01] + assign[W24_P02] + assign[W24_P04]
   + assign[W24_P05] + assign[W24_P06] + assign[W24_P07] + assign[W24_P08]
   + assign[W24_P09] <= 1
 R20: assign[W16_P00] + assign[W16_P01] + assign[W16_P03] + assign[W16_P04]
   + assign[W16_P05] + assign[W16_P08] <= 1
 R21: <= 1
Bounds
Binaries
 assign[W02_P00] assign[W02_P01] assign[W02_P04] assign[W02_P06]
 assign[W02_P08] assign[W01_P01] assign[W01_P03] assign[W01_P04]
 assign[W01_P05] assign[W01_P06] assign[W01_P08] assign[W01_P09]
 assign[W10_P00] assign[W10_P01] assign[W10_P02] assign[W10_P05]
 assign[W10_P07] assign[W10_P08] assign[W10_P09] assign[W17_P00]
 assign[W17_P02] assign[W17_P04] assign[W17_P05] assign[W17_P06]
 assign[W17_P08] assign[W18_P00] assign[W18_P02] assign[W18_P03]
 assign[W18_P04] assign[W18_P05] assign[W18_P06] assign[W18_P08]
 assign[W18_P09] assign[W05_P00] assign[W05_P01] assign[W05_P02]
 assign[W05_P04] assign[W05_P06] assign[W05_P08] assign[W05_P09]
 assign[W16_P00] assign[W16_P01] assign[W16_P03] assign[W16_P04]
 assign[W16_P05] assign[W16_P08] assign[W00_P01] assign[W00_P02]
 assign[W00_P04] assign[W00_P05] assign[W00_P08] assign[W00_P09]
 assign[W11_P00] assign[W11_P01] assign[W11_P02] assign[W11_P03]
 assign[W11_P04] assign[W11_P06] assign[W11_P07] assign[W11_P08]
 assign[W11_P09] assign[W24_P00] assign[W24_P01] assign[W24_P02]
 assign[W24_P04] assign[W24_P05] assign[W24_P06] assign[W24_P07]
 assign[W24_P08] assign[W24_P09] assign[W14_P01] assign[W14_P02]
 assign[W14_P03] assign[W14_P04] assign[W14_P05] assign[W14_P06]
 assign[W14_P07] assign[W14_P08] assign[W14_P09]
End

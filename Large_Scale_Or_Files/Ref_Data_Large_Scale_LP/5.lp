\ Model ProductionProfit
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x8b353744fa1403b3
Maximize
  80 x1 + 125 x2
Subject To
 Steel: 8 x1 + 5 x2 <= 3500
 Iron: 6 x1 + 4 x2 <= 1800
 MachineHours: 4 x1 + 5 x2 <= 2800
Bounds
End

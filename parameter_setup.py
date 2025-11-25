# ---- Parameter setup ----
# filedir = "H:\My Drive\Research\MarPPSS"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/MarPPSS"

event_name = "S0976a_src4s"
mode = 2 # 1 - PP, 2 - SS, 3 - joint
rayp = 10.13/59.1579 # S0976a: PP 4.84 s/deg, SS 10.13 s/deg
useCD = False

PPdir = event_name+"_PP"
SSdir = event_name+"_SS"
syndir = event_name

data_type = ["PP", "SS", "joint"]
modname = event_name+"_"+data_type[mode-1]
runname = "run1_centerP"

HRange = (1, 55)
vRange = (1.0, 5.0)
rhoRange = (1.6, 2.0)
maxN = 2
stdP = 0.1

totalSteps     = int(1e6)
burnInSteps    = int(8e5)
nSaveModels    = 200
actionsPerStep = 1
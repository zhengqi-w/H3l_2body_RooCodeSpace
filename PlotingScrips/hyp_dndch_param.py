import ROOT 
import numpy as np

yield_hyp_pp_mb = 2.1e-8
yield_hyp_pp_mb_stat = 0.6e-8
yield_hyp_pp_mb_syst = 0.4e-8

yield_hyp_pp_hm = 2.4e-7
yield_hyp_pp_hm_stat = 0.5e-7
yield_hyp_pp_hm_syst = 0.3e-7

yield_hyp_pPb = 6.83e-7
yield_hyp_pPb_stat = 1.8e-7
yield_hyp_pPb_syst = 1.2e-7

yield_hyp_OO = 7.39e-7
yield_hyp_OO_stat = 0.59e-7
yield_hyp_OO_syst = 1.69e-7

yield_hyp_PbPb_0_10 = 4.83e-5
yield_hyp_PbPb_0_10_stat = 0.23e-5
yield_hyp_PbPb_0_10_syst = 0.57e-5

yield_hyp_PbPb_10_30 = 2.62e-5
yield_hyp_PbPb_10_30_stat = 0.25e-5
yield_hyp_PbPb_10_30_syst = 0.40e-5

yield_hyp_PbPb_30_50 = 1.27e-5
yield_hyp_PbPb_30_50_stat = 0.10e-5
yield_hyp_PbPb_30_50_syst = 0.14e-5

##################################################
x_oo = np.array([43.9])
x_pPb = np.array([29.4])

x_pp_MB = np.array([6.9])
x_pp_HM = np.array([30.8])

x_R2_0_10 = np.array([1764])
x_R2_10_30 = np.array([983])
x_R2_30_50 = np.array([415])

x_oo_unc = np.array([2.5])
x_pPb_unc = np.array([3])

x_pp_Run2_MB_unc = np.array([0.9])
x_pp_Run2_HM_unc = np.array([3.7])

x_R2_0_10_unc = np.array([51.6])
x_R2_10_30_unc = np.array([36.9])
x_R2_30_50_unc = np.array([19.2])

######################################
y_pp_MB = np.array([yield_hyp_pp_mb])
y_pp_HM = np.array([yield_hyp_pp_hm])
y_oo = np.array([yield_hyp_OO])
y_pPb = np.array([yield_hyp_pPb])
y_R2_0_10 = np.array([yield_hyp_PbPb_0_10])
y_R2_10_30 = np.array([yield_hyp_PbPb_10_30])
y_R2_30_50 = np.array([yield_hyp_PbPb_30_50])

y_pp_MB_syst = np.array([yield_hyp_pp_mb_syst])
y_pp_HM_syst = np.array([yield_hyp_pp_hm_syst])

y_oo_syst = np.array([yield_hyp_OO_syst])
y_pPb_syst = np.array([yield_hyp_pPb_syst])

y_R2_0_10_syst = np.array([yield_hyp_PbPb_0_10_syst])
y_R2_10_30_syst = np.array([yield_hyp_PbPb_10_30_syst])
y_R2_30_50_syst = np.array([yield_hyp_PbPb_30_50_syst])

################################33
y_pp_MB_stat = np.array([yield_hyp_pp_mb_stat])
y_pp_HM_stat = np.array([yield_hyp_pp_hm_stat])

y_oo_stat = np.array([yield_hyp_OO_stat])
y_pPb_stat = np.array([yield_hyp_pPb_stat])

y_R2_0_10_stat = np.array([yield_hyp_PbPb_0_10_stat])
y_R2_10_30_stat = np.array([yield_hyp_PbPb_10_30_stat])
y_R2_30_50_stat = np.array([yield_hyp_PbPb_30_50_stat])

#########################################################
def make_point(x, y, yerr):
    return ROOT.TGraphErrors(
        1,
        np.array([float(x)], dtype='d'),
        np.array([float(y)], dtype='d'),
        np.array([0.0], dtype='d'),
        np.array([float(yerr)], dtype='d')
    )

def make_box(x, y, x_unc, y_syst, color):
    box = ROOT.TBox(
        x - x_unc,
        y - y_syst,
        x + x_unc,
        y + y_syst
    )
    box.SetFillStyle(0)
    box.SetLineColor(color)
    box.SetLineWidth(2)
    return box

gr_pp_MB = make_point(x_pp_MB[0], y_pp_MB[0], yield_hyp_pp_mb_stat)
gr_pp_HM = make_point(x_pp_HM[0], y_pp_HM[0], yield_hyp_pp_hm_stat)
gr_oo = make_point(x_oo[0], y_oo[0], yield_hyp_OO_stat)
gr_pPb = make_point(x_pPb[0], y_pPb[0], yield_hyp_pPb_stat)
gr_R2_0_10 = make_point(x_R2_0_10[0], y_R2_0_10[0], yield_hyp_PbPb_0_10_stat)
gr_R2_10_30 = make_point(x_R2_10_30[0], y_R2_10_30[0], yield_hyp_PbPb_10_30_stat)
gr_R2_30_50 = make_point(x_R2_30_50[0], y_R2_30_50[0], yield_hyp_PbPb_30_50_stat)

boxes = [
    make_box(x_pp_MB[0], y_pp_MB[0], x_pp_Run2_MB_unc[0], y_pp_MB_syst[0], ROOT.kP8Cyan),
    make_box(x_pp_HM[0], y_pp_HM[0], x_pp_Run2_HM_unc[0], y_pp_HM_syst[0], ROOT.kP8Cyan),
    
    make_box(x_pPb[0], y_pPb[0], x_pPb_unc[0], y_pPb_syst[0], ROOT.kAzure-4),

    make_box(x_oo[0], y_oo[0], x_oo_unc[0], y_oo_syst[0], ROOT.kBlue+2),
    
    make_box(x_R2_0_10[0], y_R2_0_10[0], x_R2_0_10_unc[0], y_R2_0_10_syst[0], ROOT.kP10Blue),
    make_box(x_R2_10_30[0], y_R2_10_30[0], x_R2_10_30_unc[0], y_R2_10_30_syst[0], ROOT.kP10Blue),
    make_box(x_R2_30_50[0], y_R2_30_50[0], x_R2_30_50_unc[0], y_R2_30_50_syst[0], ROOT.kP10Blue),
]

boxes_s = [
    make_box(x_pp_MB[0], y_pp_MB[0], x_pp_Run2_MB_unc[0], y_pp_MB_syst[0], ROOT.kP8Cyan),
    make_box(x_pp_HM[0], y_pp_HM[0], x_pp_Run2_HM_unc[0], y_pp_HM_syst[0], ROOT.kP8Cyan),
    
    make_box(x_pPb[0], y_pPb[0], x_pPb_unc[0], y_pPb_syst[0], ROOT.kP8Azure),
    
    make_box(x_oo[0], y_oo[0], x_oo_unc[0], y_oo_syst[0], ROOT.kBlue+2),
]

for g, c, m in [
    (gr_pp_MB, ROOT.kP8Cyan, 20),
    (gr_pp_HM, ROOT.kP8Cyan, 22),
    # (gr_pp_Run3, ROOT.kBlack, 25),
    (gr_pPb, ROOT.kAzure-4, 23),
    (gr_oo, ROOT.kBlue+2, 33),
    # (gr_R1_0_10, ROOT.kBlack, 28),
    # (gr_R1_10_50, ROOT.kBlack, 28),
    (gr_R2_0_10, ROOT.kP10Blue, 21),
    (gr_R2_10_30, ROOT.kP10Blue, 21),
    (gr_R2_30_50, ROOT.kP10Blue, 21),
]:
    g.SetMarkerColor(c)
    g.SetLineColor(c)
    g.SetMarkerStyle(m)
    g.SetMarkerSize(1)

    x = np.array([
    x_pp_MB[0],
    x_pPb[0],
    x_oo[0],
    x_R2_30_50[0],
    x_R2_10_30[0],
    x_R2_0_10[0]
], dtype='d')

y = np.array([
    y_pp_MB[0],
    y_pPb[0],
    y_oo[0],
    y_R2_30_50[0],
    y_R2_10_30[0],
    y_R2_0_10[0]
], dtype='d')

# yerr = np.array([
#     np.sqrt(y_pp_MB_stat[0]**2      + y_pp_MB_syst[0]**2),
#     np.sqrt(y_pPb_stat[0]**2        + y_pPb_syst[0]**2),
#     np.sqrt(y_oo_stat[0]**2         + y_oo_syst[0]**2),
#     np.sqrt(y_R2_30_50_stat[0]**2  + y_R2_30_50_syst[0]**2),
#     np.sqrt(y_R2_10_30_stat[0]**2  + y_R2_10_30_syst[0]**2),
#     np.sqrt(y_R2_0_10_stat[0]**2   + y_R2_0_10_syst[0]**2)
# ], dtype='d')

# yerr = np.array([
#     np.sqrt(y_pp_MB_stat[0]**2),
#     np.sqrt(y_pPb_stat[0]**2),
#     np.sqrt(y_oo_stat[0]**2 ),
#     np.sqrt(y_R2_30_50_stat[0]**2),
#     np.sqrt(y_R2_10_30_stat[0]**2),
#     np.sqrt(y_R2_0_10_stat[0]**2)
# ], dtype='d')

# yerr = np.array([
#     y_pp_MB_stat[0],
#     y_pPb_stat[0],
#     y_oo_stat[0],
#     y_R2_30_50_stat[0],
#     y_R2_10_30_stat[0],
#     y_R2_0_10_stat[0]
# ], dtype='d')


xerr = np.zeros(len(x), dtype='d')
yerr = np.zeros(len(x), dtype='d')

gr_fit = ROOT.TGraphErrors(
    len(x),
    x,
    y,
    xerr,
    yerr
)

#gr_fit = ROOT.TGraph(len(x), x, y)
    
c = ROOT.TCanvas("c","c",3000,2400)
c.SetLogy()

frame = c.DrawFrame(4, 5e-9, 2500, 1.0e-3)
frame.GetXaxis().SetTitleFont(42)
frame.GetXaxis().SetTitle("#LTd#it{N}_{ch}/d#it{#eta}#GT_{|#it{#eta}|<0.5}")
frame.GetYaxis().SetTitle("#LTd#it{N}/d#it{y}#GT")

for b in boxes:
    b.Draw("same")
    
for g in [gr_pp_MB, gr_pp_HM, gr_pPb, gr_oo,
          gr_R2_0_10, gr_R2_10_30, gr_R2_30_50]:
    g.Draw("PEZ SAME")
    
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(43)
latex.SetTextSize(0.07)
latex.DrawLatex(0.18, 0.85, "ALICE")

latex_2 = ROOT.TLatex()
latex_2.SetNDC()
latex_2.SetTextFont(43)
latex_2.SetTextSize(0.07)
latex_2.DrawLatex(0.75, 0.85, "{}^{3}_{#Lambda}H")

def section_title(x, y, text):
    t = ROOT.TLatex()
    t.SetNDC()
    t.SetTextFont(62)
    t.SetTextSize(0.04)
    t.DrawLatex(x, y, text)


def make_legend(x1, y1, x2, y2):
    leg = ROOT.TLegend(x1, y1, x2, y2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    return leg

x1, x2 = 0.55, 0.88   # mesma largura para todas
h = 0.08              # altura fixa
gap = 0.015           # espaço vertical
y_start = 0.78        # topo da primeira

# ================= pp =================
x_pp = np.array([1, 2, 3, 4, 5], dtype='float64')
y_pp = np.array([2, 4, 1, 3, 5], dtype='float64')

g_pp = ROOT.TGraph(len(x_pp), x_pp, y_pp)

leg_pp = make_legend(0.18, 0.72, 0.28, 0.81)
leg_pp.AddEntry(g_pp, "pp", "")
leg_pp.AddEntry(gr_pp_MB, "#sqrt{#it{s}} = 13 TeV, |#it{y}| < 0.5", "pez")
leg_pp.AddEntry(gr_pp_HM, "#sqrt{#it{s}} = 13 TeV, |#it{y}| < 0.8", "pez")
# leg_pp.AddEntry(gr_pp_Run3, "Run 3 (13.6 TeV)", "pl")
leg_pp.Draw()


# ================= pPb =================
x_Pb = np.array([1, 2, 3, 4, 5], dtype='float64')
y_Pb = np.array([2, 4, 1, 3, 5], dtype='float64')

g_Pb = ROOT.TGraph(len(x_Pb), x_Pb, y_Pb)

leg_pPb = make_legend(0.32, 0.72, 0.42, 0.78)
leg_pPb.AddEntry(g_Pb, r"p#font[122]{-}Pb", "")
leg_pPb.AddEntry(gr_pPb, "#sqrt{#it{s}_{NN}} = 5.02 TeV, 0-40%, -1 < #it{y} < 0", "pez")
leg_pPb.Draw()


# ================= OO =================
x_OO = np.array([1, 2, 3, 4, 5], dtype='float64')
y_OO = np.array([2, 4, 1, 3, 5], dtype='float64')

g_OO = ROOT.TGraph(len(x_OO), x_OO, y_OO)

leg_OO = make_legend(0.18, 0.64, 0.28, 0.7)
leg_OO.AddEntry(g_OO, "OO (Preliminary)", "")
leg_OO.AddEntry(gr_oo, "#sqrt{#it{s}_{NN}} = 5.36 TeV, 0-90%, |#it{y}| < 1", "pez")
leg_OO.Draw()


# ================= PbPb =================
x_PbPb = np.array([1, 2, 3, 4, 5], dtype='float64')
y_PbPb = np.array([2, 4, 1, 3, 5], dtype='float64')

g_PbPb = ROOT.TGraph(len(x_PbPb), x_PbPb, y_PbPb)

leg_PbPb = make_legend(0.32, 0.64, 0.42, 0.7)
leg_PbPb.AddEntry(g_PbPb, r"Pb#font[122]{-}Pb", "")
#leg_PbPb.AddEntry(gr_R1_0_10, "Run 1 (2.76 TeV)", "pl")
leg_PbPb.AddEntry(gr_R2_0_10, "#sqrt{#it{s}_{NN}} = 5.02 TeV, |#it{y}| < 0.5", "pez")
leg_PbPb.Draw()



f_pow = ROOT.TF1("f_lin", "[0]*TMath::Power(x,[1])", 4, 2500)
f_pow.SetParameters(3e-8, 0.)
#f_lin.SetParameter(1, 0.5)

f_pow.SetLineColor(ROOT.kBlack)
f_pow.SetLineWidth(1)

gr_fit.Fit(f_pow, "R0SY+")
f_pow.Draw("same")

n = f_pow.GetParameter(1)
n_err = f_pow.GetParError(1)

leg_fit = ROOT.TLegend(0.55,0.20,0.85,0.4)
leg_fit.SetBorderSize(0)
leg_fit.SetFillStyle(0)
leg_fit.SetTextSize(0.035)

leg_fit.AddEntry(f_pow, f"a#it{{x}}^{{N}} fit (N = {n:.2f} #pm {n_err:.2f})", "l")

leg_fit.Draw()

###########################################################

c_zoom = ROOT.TCanvas("c_zoom","c_zoom",3000,2400)

frame_zoom = c_zoom.DrawFrame(4, 5e-9, 60, 1.5e-6)
frame_zoom.GetXaxis().SetTitleFont(42)
frame_zoom.GetXaxis().SetTitle("#LTd#it{N}_{ch}/d#it{#eta}#GT_{|#it{#eta}|<0.5}")
frame_zoom.GetYaxis().SetTitle("#LTd#it{N}/d#it{y}#GT")

for b_s in boxes_s:
    b_s.Draw("same")
    
for g_r in [gr_pp_MB, gr_pp_HM, gr_pPb, gr_oo]:
    g_r.Draw("PEZ SAME")
    
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(43)
latex.SetTextSize(0.07)
latex.DrawLatex(0.18, 0.85, "ALICE")

latex_2 = ROOT.TLatex()
latex_2.SetNDC()
latex_2.SetTextFont(43)
latex_2.SetTextSize(0.07)
latex_2.DrawLatex(0.75, 0.85, "{}^{3}_{#Lambda}H")

x1, x2 = 0.55, 0.88   # mesma largura para todas
h = 0.08              # altura fixa
gap = 0.015           # espaço vertical
y_start = 0.78        # topo da primeira

# ================= pp =================
x_pp = np.array([1, 2, 3, 4, 5], dtype='float64')
y_pp = np.array([2, 4, 1, 3, 5], dtype='float64')

g_pp_zoom = ROOT.TGraph(len(x_pp), x_pp, y_pp)
leg_pp_zoom = make_legend(0.18, 0.72, 0.28, 0.8)
leg_pp_zoom.AddEntry(g_pp, "pp", "")
leg_pp_zoom.AddEntry(gr_pp_MB, "#sqrt{#it{s}} = 13 TeV, |#it{y}| < 0.5", "pez")
leg_pp_zoom.AddEntry(gr_pp_HM, "#sqrt{#it{s}} = 13 TeV, |#it{y}| < 0.8", "pez")
# leg_pp.AddEntry(gr_pp_Run3, "Run 3 (13.6 TeV)", "pl")
leg_pp_zoom.Draw()


# ================= pPb =================
x_Pb = np.array([1, 2, 3, 4, 5], dtype='float64')
y_Pb = np.array([2, 4, 1, 3, 5], dtype='float64')

g_Pb_zoom = ROOT.TGraph(len(x_Pb), x_Pb, y_Pb)

leg_pPb_zoom = make_legend(0.32, 0.72, 0.42, 0.78)
leg_pPb_zoom.AddEntry(g_Pb, r"p#font[122]{-}Pb", "")
leg_pPb_zoom.AddEntry(gr_pPb, "#sqrt{#it{s}_{NN}} = 5.02 TeV, 0-40%, -1 < #it{y} < 0", "pez")
leg_pPb_zoom.Draw()


# ================= OO =================
x_OO = np.array([1, 2, 3, 4, 5], dtype='float64')
y_OO = np.array([2, 4, 1, 3, 5], dtype='float64')

g_OO_zoom = ROOT.TGraph(len(x_OO), x_OO, y_OO)

leg_OO_zoom = make_legend(0.18, 0.64, 0.28, 0.7)
leg_OO_zoom.AddEntry(g_OO, "OO (Preliminary)", "")
leg_OO_zoom.AddEntry(gr_oo, "#sqrt{#it{s}_{NN}} = 5.36 TeV, 0-90%, |#it{y}| < 1", "pez")
leg_OO_zoom.Draw()

leg_fit_zoom = ROOT.TLegend(0.55,0.20,0.85,0.30)
leg_fit_zoom.SetBorderSize(0)
leg_fit_zoom.SetFillStyle(0)
leg_fit_zoom.AddEntry(f_lin, "Linear parametrization: ax + b", "l")
#leg_fit_zoom.AddEntry(f_pow, f"ax^{{N}} + b, N = {n:.2f} #pm {n_err:.2f}", "l")

leg_fit_zoom.Draw()

f_lin.Draw("same")


output = ROOT.TFile("3HL_dndch_param.root", "RECREATE")       
c.Write()
c_zoom.Write()


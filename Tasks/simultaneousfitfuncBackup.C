/* inline FitResult ProcessSeprateFit(std::vector<TH1*> hists, double ctmin, double ctmax) {
    FitResult res;
    res.tao = 0.0;
    res.taoErr = 0.0;
    res.chi2 = 0.0;
    res.ndf = 0;
    res.chi2PerChannel.clear();
    res.ndfPerChannel.clear();
    res.fitframes.clear();

    // Observable (shared object used to build per-channel frames)
    RooRealVar ct("ct","decay length", ctmin, ctmax);

    const double conv = 0.0299792458; // c in cm / ps factor used previously

    // loop channels and fit each histogram independently with an exponential
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        int nb = h->GetNbinsX();
        if (nb <= 0) continue;

        // unique names per channel
        std::string idx = std::to_string(i);
        std::string dataName = "data_" + idx;
        std::string tauName  = "tau_"  + idx;
        std::string slopeName= "slope_" + idx;
        std::string pdfName  = "exp_"  + idx;

        // create RooDataHist from TH1
        RooDataHist dataHist(dataName.c_str(), dataName.c_str(), RooArgList(ct), h);

        // per-channel lifetime parameter
        double initTau = 253.0;
        RooRealVar tau(tauName.c_str(), "lifetime", initTau, 1.0, 20000.0);

        // slope = -1/(c * tau)
        std::string formula = "-1.0/(@0*" + std::to_string(conv) + ")";
        RooFormulaVar slope( slopeName.c_str(), formula.c_str(), RooArgList(tau));

        // exponential PDF for this channel
        RooExponential exp(pdfName.c_str(), pdfName.c_str(), ct, slope);

        // fit PDF to this channel's data (use SumW2Error to account for histogram errors)
        RooFitResult* fitRes = exp.fitTo(dataHist, Save(true), SumW2Error(kTRUE), PrintLevel(-1));
        if (!fitRes) {
            std::cerr << "Fit failed for channel " << i << "\n";
        }

        // build a histogram of the fitted PDF with same binning as data
        TH1* hpdf = exp.createHistogram(Form("hpdf_%s", idx.c_str()), ct,
                                       Binning(nb, h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax()));
        if (!hpdf) {
            std::cerr << "Failed to create pdf histogram for channel " << i << "\n";
            continue;
        }

        // scale pdf histogram to the same integral as data histogram
        double integral_pdf = hpdf->Integral();
        double integral_data = h->Integral();
        if (integral_pdf > 0) hpdf->Scale(integral_data / integral_pdf);

        // compute chi2 between data and pdf histogram
        double chi2 = 0.0;
        int nBinsUsed = 0;
        for (int b = 1; b <= nb; ++b) {
            double o = h->GetBinContent(b);
            double e = hpdf->GetBinContent(b);
            double err = h->GetBinError(b);
            if (err <= 0) err = std::sqrt(std::max(1.0, o)); // fallback
            if (o == 0.0 && e == 0.0) continue;
            chi2 += (o - e)*(o - e) / (err * err);
            ++nBinsUsed;
        }
        int ndf = std::max(0, nBinsUsed - 1); // one fitted parameter (tau) per channel

        // prepare frame: data and fitted pdf overlaid
        RooPlot* frame = ct.frame(Title(Form("Channel %zu", i)));
        dataHist.plotOn(frame, MarkerStyle(20));
        exp.plotOn(frame, LineColor(kRed), LineWidth(2));
        // annotate fit result on the frame
        double fittedTau = tau.getVal();
        double fittedTauErr = tau.getError();
        TPaveText* pt = new TPaveText(0.55, 0.7, 0.9, 0.9, "NDC");
        pt->SetFillStyle(0);
        pt->SetBorderSize(0);
        pt->AddText(Form("tau = %.3f #pm %.3f", fittedTau, fittedTauErr));
        pt->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
        frame->addObject(pt);

        // store per-channel results
        res.chi2PerChannel.push_back(chi2);
        res.ndfPerChannel.push_back(ndf);
        res.fitframes.push_back(frame);

        // accumulate global chi2/ndf
        res.chi2 += chi2;
        res.ndf  += ndf;

        // cleanup temporary histogram (RooPlot owns drawn objects; keep hpdf allocated to avoid double free)
        // note: hpdf remains in memory; ROOT will manage when program exits or file closed
    }
    // leave tao/taoErr as zero (not applicable for per-channel independent fits)
    return res;
}

inline FitResult ProcessSimultaneousFitSameBin(std::vector<TH1*> hists,double ctmin, double ctmax)
{
    FitResult res;
    res.tao = 0.0;
    res.taoErr = 0.0;
    res.chi2 = 0.0;
    res.ndf = 0;
    res.chi2PerChannel.clear();
    res.ndfPerChannel.clear();
    res.fitframes.clear();

    if (hists.empty()) return res;

    // -------------------------------
    // Observable and category
    // -------------------------------
    RooRealVar ct("ct", "decay length", ctmin, ctmax);
    RooCategory ch("ch", "channel");

    std::vector<std::string> labels;
    for (size_t i=0;i<hists.size();++i) {
        std::string lab = "ch" + std::to_string(i);
        ch.defineType(lab.c_str());
        labels.push_back(lab);
    }

    // -------------------------------
    // Build per-channel RooDataHist (binned data)
    // -------------------------------
    std::vector<RooDataHist*> dhists(hists.size(), nullptr);
    for (size_t i = 0; i < hists.size(); ++i) {
        if (!hists[i]) continue;
        dhists[i] = new RooDataHist(Form("dh_%zu", i),
                                    "datahist",
                                    RooArgList(ct),
                                    hists[i]);
    }

    // ensure dhists built and labels defined above
    if (std::all_of(dhists.begin(), dhists.end(), [](RooDataHist* p){ return p == nullptr; })) return res;
    std::vector<size_t> validIdx;
    for (size_t i = 0; i < dhists.size(); ++i) {
        if (dhists[i]) validIdx.push_back(i);
    }
    if (validIdx.empty()) return res;

    // Helper lambdas to get label and dhist reference
    auto L = [&](size_t k)->const char* { return labels[k].c_str(); };
    auto D = [&](size_t k)->RooDataHist& { return *dhists[k]; };
    RooDataHist* comb_ptr = nullptr;
    size_t nvalid = validIdx.size();

    // Support up to 7 imports by explicit overload expansion
    if (nvalid <= 7) {
        switch (nvalid) {
            case 1:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])));
                break;
            case 2:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])));
                break;
            case 3:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])),
                                           Import(L(validIdx[2]), D(validIdx[2])));
                break;
            case 4:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])),
                                           Import(L(validIdx[2]), D(validIdx[2])),
                                           Import(L(validIdx[3]), D(validIdx[3])));
                break;
            case 5:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])),
                                           Import(L(validIdx[2]), D(validIdx[2])),
                                           Import(L(validIdx[3]), D(validIdx[3])),
                                           Import(L(validIdx[4]), D(validIdx[4])));
                break;
            case 6:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])),
                                           Import(L(validIdx[2]), D(validIdx[2])),
                                           Import(L(validIdx[3]), D(validIdx[3])),
                                           Import(L(validIdx[4]), D(validIdx[4])),
                                           Import(L(validIdx[5]), D(validIdx[5])));
                break;
            case 7:
                comb_ptr = new RooDataHist("comb", "combined binned data",
                                           RooArgList(ct),
                                           Index(ch),
                                           Import(L(validIdx[0]), D(validIdx[0])),
                                           Import(L(validIdx[1]), D(validIdx[1])),
                                           Import(L(validIdx[2]), D(validIdx[2])),
                                           Import(L(validIdx[3]), D(validIdx[3])),
                                           Import(L(validIdx[4]), D(validIdx[4])),
                                           Import(L(validIdx[5]), D(validIdx[5])),
                                           Import(L(validIdx[6]), D(validIdx[6])));
                break;
            default:
                break;
        }
    } else {
        // fallback: construct empty comb with Index(ch) then add channels one-by-one
        comb_ptr = new RooDataHist("comb", "combined binned data", RooArgList(ct), Index(ch));
        for (size_t ii = 0; ii < validIdx.size(); ++ii) {
            size_t k = validIdx[ii];
            comb_ptr->add(*dhists[k], labels[k].c_str());
        }
    }

    if (!comb_ptr) {
        std::cerr << "[Error] Failed to build combined RooDataHist (comb_ptr == nullptr)\n";
        return res;
    }
    RooDataHist& comb = *comb_ptr;
    // comb is now populated with the per-channel histograms from imports
    // -------------------------------
    // Shared lifetime parameter
    // -------------------------------
    RooRealVar tau("tau", "lifetime", 253.0, 150, 260);
    RooFormulaVar slope("slope", "-1.0/(@0*0.0299792458)", RooArgList(tau));
    // -------------------------------
    // Per-channel PDFs + yields (extended)
    // -------------------------------
    std::vector<RooAbsPdf*> pdfs;
    std::vector<RooRealVar*> nvars;
    std::vector<RooExtendPdf*> extends;
    pdfs.reserve(hists.size());
    nvars.reserve(hists.size());
    extends.reserve(hists.size());
    for (size_t i=0;i<hists.size();++i) {
        std::string idx = std::to_string(i);

        RooExponential* exp = new RooExponential(
            ("exp_"+idx).c_str(),
            ("exp_"+idx).c_str(),
            ct, slope);

        pdfs.push_back(exp);

        double initY = std::max(0.0, hists[i] ? hists[i]->Integral() : 0.0);
        RooRealVar* nvar = new RooRealVar(
            ("n_"+idx).c_str(),
            ("n_"+idx).c_str(),
            initY, 0.0, std::max(1.0, initY*1e3+1.0));

        nvars.push_back(nvar);

        RooExtendPdf* ext = new RooExtendPdf(
            ("ext_"+idx).c_str(),
            ("ext_"+idx).c_str(),
            *exp, *nvar);

        extends.push_back(ext);
    }
    // -------------------------------
    // Simultaneous PDF
    // -------------------------------
    RooSimultaneous sim("sim", "simultaneous pdf", ch);
    for (size_t i=0;i<extends.size();++i)
        sim.addPdf(*extends[i], labels[i].c_str());
    RooChi2Var chi2Var("chi2Var", "chi2Var", sim, comb);
    RooMinimizer minim(chi2Var);
    minim.setPrintLevel(-1);
    minim.setStrategy(1);
    minim.minimize("Minuit2", "migrad");
    minim.hesse(); // improve error estimates
    RooFitResult* rr = minim.save();

    if (rr) {
        res.tao = tau.getVal();
        res.taoErr = tau.getError();
    }

    // Compute per-channel chi2/ndf using the fitted parameters (integral expectation),
    // but take global chi2 from RooChi2Var
    res.chi2 = chi2Var.getVal();

    // Count total used bins and per-channel contributions for ndf calculation
    int totalBinsUsed = 0;
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        int nb = h->GetNbinsX();
        if (nb <= 0) continue;

        RooAbsPdf* pdf = pdfs[i];
        RooRealVar* nvar = nvars[i];
        double fitY = nvar->getVal();

        double chi2_ch = 0.0;
        int binsUsed = 0;
        for (int b = 1; b <= nb; ++b) {
            double xlow  = h->GetXaxis()->GetBinLowEdge(b);
            double xhigh = h->GetXaxis()->GetBinUpEdge(b);
            ct.setRange("binrange", xlow, xhigh);
            std::unique_ptr<RooAbsReal> integ(pdf->createIntegral(RooArgSet(ct), Range("binrange")));
            double p = integ->getVal();
            double expect = fitY * p;

            double obs = h->GetBinContent(b);
            double err = h->GetBinError(b);
            if (err <= 0) err = std::sqrt(std::max(1.0, obs));

            if (obs == 0.0 && expect == 0.0) continue;
            chi2_ch += (obs - expect) * (obs - expect) / (err * err);
            ++binsUsed;
        }
        int ndf_ch = std::max(0, binsUsed - 1); // one free parameter per channel is the yield (tau is global)
        res.chi2PerChannel.push_back(chi2_ch);
        res.ndfPerChannel.push_back(ndf_ch);

        res.chi2 += 0.0; // global chi2 already set from RooChi2Var
        res.ndf += ndf_ch;
        totalBinsUsed += binsUsed;
        // collect bin edges
        std::vector<double> edges;
        edges.reserve(nb + 1);
        for (int b = 1; b <= nb; ++b) {
            edges.push_back(h->GetXaxis()->GetBinLowEdge(b));
        }
        // add upper edge of last bin
        edges.push_back(h->GetXaxis()->GetBinLowEdge(nb) + h->GetXaxis()->GetBinWidth(nb));
        double xmin = edges.front();
        double xmax = edges.back();
        if (xmin == xmax) xmax = xmin + 1e-6; // guard

        // build RooBinning with exact bin boundaries (preserves irregular bins and order)
        RooBinning rb;
        for (double e : edges) rb.addBoundary(e);

        // create RooDataHist from TH1 and a frame, then plot using the RooBinning
        RooDataHist dh(Form("dh_print_%zu", i), "datahist", RooArgList(ct), h);
        RooPlot* frame = ct.frame(); // use default frame
        dh.plotOn(frame, Binning(rb), DataError(RooAbsData::SumW2)); // use exact bin edges and proper errors
        // overlay pdf (create histogram with same binning to match scaling)
        TH1* hpdf = pdf->createHistogram(Form("hpdf_plot_%zu", i), ct, Binning(rb));
        if (hpdf) {
            double sf = (h->Integral() > 0 && hpdf->Integral() > 0) ? (h->Integral() / hpdf->Integral()) : 1.0;
            hpdf->Scale(sf);
            // convert to RooCurve on the same frame by plotting pdf itself (will respect the same variable range)
            pdf->plotOn(frame, LineColor(kRed), LineWidth(2));
            // optionally draw hpdf separately on the canvas when you draw the frame, but frame already has pdf curve
        } else {
            pdf->plotOn(frame, LineColor(kRed), LineWidth(2));
        }

        TPaveText* pt = new TPaveText(0.55, 0.7, 0.9, 0.9, "NDC");
        pt->SetFillStyle(0); pt->SetBorderSize(0);
        pt->AddText(Form("tau = %.3f ± %.3f", res.tao, res.taoErr));
        pt->AddText(Form("n = %.1f", nvars[i]->getVal()));
        pt->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2_ch, ndf_ch));
        frame->addObject(pt);
        res.fitframes.push_back(frame);
    }

    // derive global ndf: total bins used - number of free parameters (global tau + all yields)
    int nparams = 1 + static_cast<int>(nvars.size());
    res.ndf = std::max(0, totalBinsUsed - nparams);
    // note: res.chi2 already taken from RooChi2Var; res.ndf updated above
    return res;
} */

inline FitResult ProcessSimultaneousFit(std::vector<TH1*> hists,double ctmin, double ctmax) {
    FitResult res;
    res.tao = 0.0;
    res.taoErr = 0.0;
    res.chi2 = 0.0;
    res.ndf = 0;
    res.chi2PerChannel.clear();
    res.ndfPerChannel.clear();
    res.fitframes.clear();
    if (hists.empty()) return res;
    RooRealVar ct("ct", "decay length", ctmin, ctmax);
    RooCategory ch("ch", "channel");
    std::vector<std::string> labels;
    for (size_t i = 0; i < hists.size(); ++i) {
        std::string lab = "ch" + std::to_string(i);
        ch.defineType(lab.c_str());
        labels.push_back(lab);
    }
    // Build combined RooDataSet by iterating original TH1s and adding weighted entries.
    // Use RooArgSet(ct, ch) so we can set the category index before adding each row.
    RooDataSet* comb_ptr = new RooDataSet("comb", "combined", RooArgSet(ct, ch));
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        int nb = h->GetNbinsX();
        // set category index to the same index defined earlier by ch.defineType(...)
        int catIndex = static_cast<int>(i);
        for (int b = 1; b <= nb; ++b) {
            double w = h->GetBinContent(b);
            if (w <= 0) continue;
            double center = h->GetXaxis()->GetBinCenter(b);
            ct.setVal(center);
            ch.setIndex(catIndex); // set category value before adding
            RooArgSet row(ct, ch);
            comb_ptr->add(row, w);
        }
    }
    if (!comb_ptr) {
        std::cerr << "[Error] failed to build combined RooDataSet\n";
        return res;
    }
    RooDataSet& comb = *comb_ptr;
    RooRealVar tau("tau", "lifetime", 253.0, 150, 260);
    RooFormulaVar slope("slope", "-1.0/(@0*0.0299792458)", RooArgList(tau));
    std::vector<RooAbsPdf*> pdfs;
    std::vector<RooRealVar*> nvars;
    std::vector<RooExtendPdf*> extends;
    pdfs.reserve(hists.size());
    nvars.reserve(hists.size());
    extends.reserve(hists.size());
    for (size_t i=0;i<hists.size();++i) {
        std::string idx = std::to_string(i);

        RooExponential* exp = new RooExponential(
            ("exp_"+idx).c_str(),
            ("exp_"+idx).c_str(),
            ct, slope);

        pdfs.push_back(exp);

        double initY = std::max(0.0, hists[i] ? hists[i]->Integral() : 0.0);
        RooRealVar* nvar = new RooRealVar(
            ("n_"+idx).c_str(),
            ("n_"+idx).c_str(),
            initY, 80, std::max(1.0, initY*1e3+1.0));

        nvars.push_back(nvar);

        RooExtendPdf* ext = new RooExtendPdf(
            ("ext_"+idx).c_str(),
            ("ext_"+idx).c_str(),
            *exp, *nvar);

        extends.push_back(ext);
    }
    RooSimultaneous sim("sim", "simultaneous pdf", ch);
    for (size_t i=0;i<extends.size();++i) sim.addPdf(*extends[i], labels[i].c_str());
    RooFitResult* fitRes = sim.fitTo(
        comb,             // RooDataSet reference (combined weighted data)
        Save(true),
        Extended(kTRUE),
        PrintLevel(-1)
    );
    if (fitRes) {
        res.tao = tau.getVal();
        res.taoErr = tau.getError();
    }

    // Count total used bins and per-channel contributions for ndf calculation
    int totalBinsUsed = 0;
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        int nb = h->GetNbinsX();
        if (nb <= 0) continue;

        RooAbsPdf* pdf = pdfs[i];
        RooRealVar* nvar = nvars[i];
        double fitY = nvar->getVal();

        double chi2_ch = 0.0;
        int binsUsed = 0;
        for (int b = 1; b <= nb; ++b) {
            double xlow  = h->GetXaxis()->GetBinLowEdge(b);
            double xhigh = h->GetXaxis()->GetBinUpEdge(b);
            cout << "Bin " << b << ": xlow=" << xlow << ", xhigh=" << xhigh << "\n";
            ct.setRange("binrange", xlow, xhigh);
            std::unique_ptr<RooAbsReal> integ(pdf->createIntegral(RooArgSet(ct), Range("binrange")));
            double p = integ->getVal();
            double expect = fitY * p;
            cout << "  Integral over bin = " << p << ", expected count = " << expect << "\n";
            cout << "n  fitY = " << fitY << "\n";
            double obs = h->GetBinContent(b);
            double err = h->GetBinError(b);
            if (err <= 0) err = std::sqrt(std::max(1.0, obs));
            if (obs == 0.0 && expect == 0.0) continue;
            chi2_ch += (obs - expect) * (obs - expect) / (err * err);
            ++binsUsed;
            cout << "  obs=" << obs << ", expect=" << expect << ", err=" << err << ", partial chi2=" << chi2_ch << "\n";
        }
        double xmin = h->GetXaxis()->GetBinLowEdge(1);
        double xmax = h->GetXaxis()->GetBinUpEdge(nb);
        ct.setRange(xmin, xmax);
        cout << "binLow=" << xmin << ", binUp=" << xmax << "\n";
        int ndf_ch = std::max(0, binsUsed - 1); // one free parameter per channel is the yield (tau is global)
        res.chi2PerChannel.push_back(chi2_ch);
        res.ndfPerChannel.push_back(ndf_ch);

        res.chi2 += chi2_ch;
        res.ndf += ndf_ch;
        totalBinsUsed += binsUsed;
        // collect bin edges
        std::vector<double> edges;
        edges.reserve(nb + 1);
        for (int b = 1; b <= nb; ++b) {
            edges.push_back(h->GetXaxis()->GetBinLowEdge(b));
        }
        // add upper edge of last bin
        edges.push_back(h->GetXaxis()->GetBinLowEdge(nb) + h->GetXaxis()->GetBinWidth(nb));
        if (xmin == xmax) xmax = xmin + 1e-6; // guard

        // build RooBinning with exact bin boundaries (preserves irregular bins and order)
        RooBinning rb;
        for (double e : edges) rb.addBoundary(e);

        // create RooDataHist from TH1 and a frame, then plot using the RooBinning
        RooDataHist dh(Form("dh_print_%zu", i), "datahist", RooArgList(ct), h);
        RooPlot* frame = ct.frame(); // use default frame
        dh.plotOn(frame, Binning(rb), DataError(RooAbsData::SumW2)); // use exact bin edges and proper errors
        // overlay pdf (create histogram with same binning to match scaling)
        TH1* hpdf = pdf->createHistogram(Form("hpdf_plot_%zu", i), ct, Binning(rb));
        if (hpdf) {
            double sf = (h->Integral() > 0 && hpdf->Integral() > 0) ? (h->Integral() / hpdf->Integral()) : 1.0;
            hpdf->Scale(sf);
            // convert to RooCurve on the same frame by plotting pdf itself (will respect the same variable range)
            pdf->plotOn(frame, LineColor(kRed), LineWidth(2));
            // optionally draw hpdf separately on the canvas when you draw the frame, but frame already has pdf curve
        } else {
            pdf->plotOn(frame, LineColor(kRed), LineWidth(2));
        }

        TPaveText* pt = new TPaveText(0.55, 0.7, 0.9, 0.9, "NDC");
        pt->SetFillStyle(0); pt->SetBorderSize(0);
        pt->AddText(Form("tau = %.3f ± %.3f", res.tao, res.taoErr));
        pt->AddText(Form("n = %.1f", nvars[i]->getVal()));
        pt->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2_ch, ndf_ch));
        frame->addObject(pt);
        res.fitframes.push_back(frame);
    }

    // derive global ndf: total bins used - number of free parameters (global tau + all yields)
    int nparams = 1 + static_cast<int>(nvars.size());
    res.ndf = std::max(0, totalBinsUsed - nparams);
    // note: res.chi2 already taken from RooChi2Var; res.ndf updated above
    return res;
}

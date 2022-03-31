#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"

#include <gsl/gsl>
#include "TSystemDirectory.h"
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"

#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using CompClusterExt = o2::itsmft::CompClusterExt;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

void checkTrackClusters()
{
    bool useITSonly = true;
    std::string itsOnlyStr = useITSonly ? "ITS-SA" : "ITS-TPC";

    double ptmax = 5;
    double ptbins = ptmax / 0.033;

    int clsize_min = 0;
    int clsize_max = 80;
    std::vector<TH2D *> histsPt(7);
    std::vector<TH2D *> histsPz(7);
    std::vector<TH2D *> histsPt0(7);
    std::vector<TH2D *> histsPt1(7);
    std::vector<TH2D *> histsPt2(7);
    std::vector<TH1D *> histsPtMean_EtaUnder4(7); // |eta| < 0.4
    std::vector<TH1D *> histsPtMean_EtaUnder8(7); // 0.4 < |eta| < 0.8
    std::vector<TH1D *> histsPtMean_EtaOver8(7);  // |eta| > 0.8
    std::vector<TH2D *> histsEta(7);
    std::vector<TH1D *> histsClSize(7);

    for (int layer{0}; layer < 7; layer++)
    {
        std::ostringstream str;
        str << "Cluster Size for L" << layer;
        std::string histsName = str.str();
        std::string pString = Form("#it{p}_{T}^{%s}", itsOnlyStr.data());

        histsClSize[layer] = new TH1D((histsName).data(), ("; " + histsName + "; Cluster size; Counts").data(), 99, 1, 100);
        histsPt[layer] = new TH2D((histsName + Form("vs pT CL size%i", clsize_max)).data(), ("; " + histsName + Form("; %s (GeV/#it{c}); Counts", pString.data())).data(), (clsize_max - clsize_min), clsize_min, clsize_max, ptbins, 0, ptmax);
        histsPz[layer] = new TH2D((histsName + Form("vs pz CL size%i", clsize_max)).data(), ("; " + histsName + Form("; %s (GeV/#it{c}); Counts", pString.data())).data(), (clsize_max - clsize_min), clsize_min, clsize_max, ptbins, 0, ptmax);
        histsPt0[layer] = new TH2D((histsName + "vs pT 0").data(), ("; " + histsName + Form("; %s (GeV/#it{c}); Counts", pString.data())).data(), 14, 1, 15, 30, 0, 1);
        histsPt1[layer] = new TH2D((histsName + "vs pT 1").data(), ("; " + histsName + Form("; %s (GeV/#it{c}); Counts", pString.data())).data(), 14, 1, 15, 30, 0, 1);
        histsPt2[layer] = new TH2D((histsName + "vs pT 2").data(), ("; " + histsName + Form("; %s (GeV/#it{c}); Counts", pString.data())).data(), 14, 1, 15, 30, 0, 1);
        histsPtMean_EtaUnder4[layer] = new TH1D((histsName + "mean vs pt (eta < 0.4)").data(), (Form("; %s (GeV/#it{c}) ; Average ", pString.data()) + histsName + "; Counts").data(), 30, 0, 1);
        histsPtMean_EtaUnder8[layer] = new TH1D((histsName + "mean vs pt (0.4 < eta < 0.8)").data(), (Form("; %s (GeV/#it{c}) ; Average ", pString.data()) + histsName + "; Counts").data(), 30, 0, 1);
        histsPtMean_EtaOver8[layer] = new TH1D((histsName + "mean vs pt (eta > 0.8)").data(), (Form("; %s (GeV/#it{c}) ; Average ", pString.data()) + histsName + "; Counts").data(), 30, 0, 1);

        // eta
        histsEta[layer] = new TH2D((histsName + Form("vs eta CL size%i", clsize_max)).data(), ("; " + histsName + "; #eta; counts").data(), (clsize_max - clsize_min), clsize_min, clsize_max, 40, -2, 2);

        // style
        histsPtMean_EtaUnder4[layer]->SetMarkerStyle(kOpenCircle);
        histsPtMean_EtaUnder4[layer]->SetMarkerColor(kRed + 1);
        histsPtMean_EtaUnder4[layer]->SetLineStyle(9);
        histsPtMean_EtaUnder4[layer]->SetLineColor(kRed + 1);

        histsPtMean_EtaUnder8[layer]->SetMarkerStyle(kOpenDiamond);
        histsPtMean_EtaUnder8[layer]->SetMarkerColor(kOrange + 2);
        histsPtMean_EtaUnder8[layer]->SetLineStyle(8);
        histsPtMean_EtaUnder8[layer]->SetLineColor(kOrange + 2);

        histsPtMean_EtaOver8[layer]->SetMarkerStyle(kOpenCross);
        histsPtMean_EtaOver8[layer]->SetMarkerColor(kSpring + 2);
        histsPtMean_EtaOver8[layer]->SetLineStyle(5);
        histsPtMean_EtaOver8[layer]->SetLineColor(kSpring + 2);

        histsClSize[layer]->SetMarkerStyle(kOpenCross);
        histsClSize[layer]->SetMarkerColor(kRed + 2);
        // histsClSize[layer]->SetLineStyle(5);
        histsClSize[layer]->SetLineColor(kRed + 2);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    std::string path = "/data/fmazzasc/its_data/505673";
    TSystemDirectory dir("MyDir", path.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs;
    for (auto fileObj : *files)
    {
        std::string file = ((TSystemFile *)fileObj)->GetName();
        if (file.substr(0, 6) == "o2_ctf")
            dirs.push_back(file);
    }

    std::sort(dirs.begin(), dirs.end());


    int counter = 0;

    for (auto &dir : dirs)
    {
        // if (counter > 0)
            // continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;

        std::string o2match_itstpc_file = path + "/" + dir + "/" + "o2match_itstpc.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "o2trac_its.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "o2clus_its.root";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

        if (!fITS || !fITSTPC || !fITSclus)
            continue;

        auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
        auto treeITS = (TTree *)fITS->Get("o2sim");
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        // Tracks
        std::vector<o2::dataformats::TrackTPCITS> *TPCITStracks = nullptr;
        std::vector<o2::its::TrackITS> *ITStracks = nullptr;

        std::vector<int> *ITSTrackClusIdx = nullptr;

        // Clusters
        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        // Setting branches
        treeITS->SetBranchAddress("ITSTrack", &ITStracks);
        treeITSTPC->SetBranchAddress("TPCITS", &TPCITStracks);

        treeITS->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);
        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

        auto &loopTree = useITSonly ? treeITS : treeITSTPC;

        for (int frame = 0; frame < loopTree->GetEntriesFast(); frame++)
        {

            if (!loopTree->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame))
                continue;

            auto pattIt = ITSpatt->cbegin();

            int trackSize = useITSonly ? ITStracks->size() : TPCITStracks->size();

            for (int iTrack{0}; iTrack < trackSize; ++iTrack)
            {

                int trackPos = useITSonly ? iTrack : int(TPCITStracks->at(iTrack).getRefITS());
                auto &ITStrack = ITStracks->at(trackPos);

                o2::track::TrackParCov baseTrack(useITSonly ? (o2::track::TrackParCov)ITStrack : TPCITStracks->at(iTrack));

                std::array<CompClusterExt, 7> TrackClus;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus[layer] = clus;
                }
                

                for (int layer{0}; layer < 7; layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {

                        auto pattID = TrackClus[layer].getPatternID();
                        int npix;
                        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID))
                        {
                            o2::itsmft::ClusterPattern patt(pattIt);
                            npix = patt.getNPixels();
                        }
                        else
                        {

                            npix = mdict.getNpixels(pattID);
                        }

                        histsClSize[layer]->Fill(npix);

                        histsPt[layer]->Fill(npix, baseTrack.getPt());
                        double pz = TMath::Sqrt(pow(baseTrack.getP(), 2) - pow(baseTrack.getPt(), 2));
                        histsPz[layer]->Fill(npix, pz);
                        histsEta[layer]->Fill(npix, baseTrack.getEta());

                        if (abs(baseTrack.getEta()) < 0.4)
                        {
                            histsPt0[layer]->Fill(npix, baseTrack.getPt());
                        }
                        else if ((abs(baseTrack.getEta()) > 0.4) && (abs(baseTrack.getEta()) < 0.8))
                        {
                            histsPt1[layer]->Fill(npix, baseTrack.getPt());
                        }
                        else
                            histsPt2[layer]->Fill(npix, baseTrack.getPt());
                    }
                }
            }
        }
        treeITS->ResetBranchAddresses();
        treeITSTPC->ResetBranchAddresses();
        treeITSclus->ResetBranchAddresses();
        fITS->Close();
        fITSTPC->Close();
        fITSclus->Close();
    }

    // mean cluster size vs. pT
    for (int layer{0}; layer < 7; layer++)
    {
        for (int ptbin = 1; ptbin < histsPt[layer]->GetNbinsY(); ptbin++)
        {
            TH1D *histPtProj0 = histsPt0[layer]->ProjectionX("histEtaProj0", ptbin, ptbin);
            TH1D *histPtProj1 = histsPt1[layer]->ProjectionX("histEtaProj1", ptbin, ptbin);
            TH1D *histPtProj2 = histsPt2[layer]->ProjectionX("histEtaProj2", ptbin, ptbin);

            histsPtMean_EtaUnder4[layer]->SetBinContent(ptbin, histPtProj0->GetMean());
            histsPtMean_EtaUnder8[layer]->SetBinContent(ptbin, histPtProj1->GetMean());
            histsPtMean_EtaOver8[layer]->SetBinContent(ptbin, histPtProj2->GetMean());
        }
    }

    auto outFile = TFile("clusITS_pass3.root", "recreate");

    // canvases
    TCanvas cClusterSize = TCanvas("cClusterSize", "cClusterSize", 1200, 800);
    TCanvas cClusterEta = TCanvas(Form("cClusterEta%i", clsize_max), Form("cClusterEta%i", clsize_max), 2200, 1200);
    TCanvas cClusterPt = TCanvas(Form("cClusterPt%i", clsize_max), Form("cClusterPt%i", clsize_max), 2200, 1200);
    TCanvas cClusterPz = TCanvas(Form("cClusterPz%i", clsize_max), Form("cClusterPz%i", clsize_max), 2200, 1200);
    TCanvas cClusterMeanVsPt = TCanvas("cClusterMeanVsPt", "cClusterMeanVsPt", 1200, 800);
    cClusterSize.Divide(4, 2);
    cClusterEta.Divide(4, 2);
    cClusterPt.Divide(4, 2);
    cClusterPz.Divide(4, 2);
    cClusterMeanVsPt.Divide(4, 2);

    // legends
    TLegend lClusterMeanVsPt = TLegend(0.2, 0.4, 0.6, 0.8);
    lClusterMeanVsPt.SetBorderSize(0);
    lClusterMeanVsPt.SetTextSize(0.055);
    lClusterMeanVsPt.AddEntry(histsPtMean_EtaUnder4[0], "|#eta| < 0.4", "lp");
    lClusterMeanVsPt.AddEntry(histsPtMean_EtaUnder8[0], "0.4 < |#eta| < 0.8", "lp");
    lClusterMeanVsPt.AddEntry(histsPtMean_EtaOver8[0], "|#eta| > 0.8", "lp");

    // latex
    TLatex laCluster;
    laCluster.SetTextSize(0.06);
    laCluster.SetNDC();
    laCluster.SetTextFont(42);

    // style
    gStyle->SetPalette(55);
    gStyle->SetPadRightMargin(0.15);
    // gStyle->SetPadLeftMargin(0.005);
    // gStyle->SetOptStat(0);
    gStyle->SetOptStat("eimrou");
    gStyle->SetStatY(0.9);
    gStyle->SetStatX(0.8);
    gStyle->SetStatW(0.4);

    for (int layer{0}; layer < 7; layer++)
    {
        auto c = cClusterSize.cd(layer + 1);
        c->SetLogy();
        histsClSize[layer]->GetYaxis()->SetDecimals();
        histsClSize[layer]->GetYaxis()->SetTitleOffset(1.);
        histsClSize[layer]->SetStats(1);
        histsClSize[layer]->DrawCopy();
        if (layer + 1 == 7)
        {
            cClusterSize.cd(8);
            laCluster.DrawLatex(0.2, 0.6, "ITS cluster size");
        }

        cClusterEta.cd(layer + 1);
        histsEta[layer]->GetYaxis()->SetDecimals();
        histsEta[layer]->GetYaxis()->SetTitleOffset(1.2);
        histsEta[layer]->GetZaxis()->SetTitleOffset(1.4);
        histsEta[layer]->DrawCopy("colz");
        if (layer + 1 == 7)
        {
            cClusterEta.cd(8);
            laCluster.DrawLatex(0.2, 0.6, "ITS cluster study vs #eta");
            laCluster.DrawLatex(0.2, 0.55, Form("(%i< CL size < %i)", clsize_min, clsize_max));
        }

        cClusterPt.cd(layer + 1);
        histsPt[layer]->GetYaxis()->SetDecimals();
        histsPt[layer]->GetYaxis()->SetTitleOffset(1.2);
        histsPt[layer]->GetZaxis()->SetTitleOffset(1.3);
        histsPt[layer]->SetStats(0);
        histsPt[layer]->DrawCopy("colz");

        cClusterPz.cd(layer + 1);
        gPad->SetFillStyle(0);
        histsPz[layer]->GetYaxis()->SetDecimals();
        histsPz[layer]->GetYaxis()->SetTitleOffset(1.2);
        histsPz[layer]->GetZaxis()->SetTitleOffset(1.4);
        histsPz[layer]->DrawCopy("colz");

        if (layer + 1 == 7)
        {
            cClusterPt.cd(8);
            laCluster.DrawLatex(0.2, 0.6, Form("%s cluster study vs #it{p}_{T}", itsOnlyStr.data()));
            laCluster.DrawLatex(0.2, 0.55, Form("(%i< CL size < %i)", clsize_min, clsize_max));
            cClusterPz.cd(8);
            laCluster.DrawLatex(0.2, 0.6, Form("%s cluster study vs #it{p}_{z}", itsOnlyStr.data()));
            laCluster.DrawLatex(0.2, 0.55, Form("(%i< CL size < %i)", clsize_min, clsize_max));
        }

        cClusterMeanVsPt.cd(layer + 1);
        histsPtMean_EtaOver8[layer]->Draw("samePL][");
        histsPtMean_EtaUnder8[layer]->Draw("samePL][");
        histsPtMean_EtaUnder4[layer]->Draw("samePL][");
        if (layer + 1 == 7)
        {
            cClusterMeanVsPt.cd(8);
            lClusterMeanVsPt.Draw("same");
        }
    }
    cClusterSize.Write();
    cClusterEta.Write();
    cClusterPt.Write();
    cClusterPz.Write();
    cClusterMeanVsPt.Write();

    auto cClSizeOB = TCanvas("cClusterSizeOB", "cClusterSizeOB");
    cClSizeOB.SetLogy();
    histsClSize[3]->GetYaxis()->SetTitle("Normalised Counts");
    histsClSize[3]->GetXaxis()->SetTitle(Form("%s Tracks Cluster Size", itsOnlyStr.data()));
    // histsClSize[3]->SetMaximum(0.4);

    histsClSize[3]->SetLineColor(kOrange);
    histsClSize[3]->DrawNormalized();
    histsClSize[4]->SetLineColor(kRed);
    histsClSize[4]->DrawNormalized("same");
    histsClSize[5]->SetLineColor(kGreen);
    histsClSize[5]->DrawNormalized("same");
    histsClSize[6]->SetLineColor(kBlue);
    histsClSize[6]->DrawNormalized("same");

    auto leg = new TLegend(0.6, 0.65, 0.8, 0.85);
    leg->SetNColumns(2);
    leg->SetMargin(0.2);
    leg->AddEntry(histsClSize[3], "L3", "l");
    leg->AddEntry(histsClSize[4], "L4", "l");
    leg->AddEntry(histsClSize[5], "L5", "l");
    leg->AddEntry(histsClSize[6], "L6", "l");
    leg->Draw();
    cClSizeOB.Write();

    cClusterEta.SaveAs(Form("ITSTPCclusterVsEta%i_%i.pdf", clsize_min, clsize_max));
    cClusterPt.SaveAs(Form("ITSTPCclusterVsPt%i_%i.pdf", clsize_min, clsize_max));
    cClusterPz.SaveAs(Form("ITSTPCclusterVsPz%i_%i.pdf", clsize_min, clsize_max));
    cClusterMeanVsPt.SaveAs("ITSClusterMeanVsPt.pdf");

    outFile.Close();
}

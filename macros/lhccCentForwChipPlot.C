#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>
#include "ReconstructionDataFormats/V0.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#include "ITSMFTReconstruction/ChipMappingITS.h"

#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include <TRandom.h>
#include <gsl/gsl>
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
#include "TPaveText.h"

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
void lhccCentForwChipPlot()
{
    gStyle->SetPalette(55);

    o2::itsmft::ChipMappingITS chipMapping;

    std::vector<TH1D *> histClPositionCent(2);
    std::vector<TH1D *> histClPositionForw(2);

    // Geometry
    o2::base::GeometryManager::loadGeometry("utils/o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "utils/ITSdictionary.bin"));

    for (int iMC{0}; iMC < 2; iMC++)
    {

        histClPositionCent[iMC] = new TH1D(Form("Average Cluster Size Central %i", iMC), "; Cluster size; Normalised counts", 99, 0.5, 99.5);
        histClPositionForw[iMC] = new TH1D(Form("Average Cluster Size Forward %i", iMC), "; Cluster size; Normalised counts", 99, 0.5, 99.5);

        std::string o2clus_its_file;
        std::string primary_vertex_file;
        if (iMC == 0)
        {
            o2clus_its_file = "505658/o2clus_its.root";
            primary_vertex_file = "505658/o2_primary_vertex.root";
        }
        else
        {
            o2clus_its_file = "ITS_MC/o2clus_its.root";
            primary_vertex_file = "ITS_MC/o2_primary_vertex.root";
        }

        auto fITSclus = TFile::Open(o2clus_its_file.data());
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");
        auto fPrimary = TFile::Open(primary_vertex_file.data());
        auto treePrimaries = (TTree *)fPrimary->Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;
        std::vector<o2::dataformats::PrimaryVertex> *Primaries = nullptr;

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
        treePrimaries->SetBranchAddress("PrimaryVertex", &Primaries);

        int nPrimaries = 0.;

        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {

            if (!treeITSclus->GetEvent(frame) || !treePrimaries->GetEvent(frame))
                continue;

            nPrimaries += Primaries->size();

            auto pattIt = ITSpatt->cbegin();

            for (auto &clus : *ITSclus)
            {
                o2::itsmft::ClusterPattern patt;

                auto chipID = clus.getChipID();
                int layer, sta, ssta, mod, chipInMod;
                chipMapping.expandChipInfoHW(chipID, layer, sta, ssta, mod, chipInMod);

                auto pattID = clus.getPatternID();
                int npix;
                if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID))
                {
                    patt.acquirePattern(pattIt);
                    npix = patt.getNPixels();
                }
                else
                {
                    npix = mdict.getNpixels(pattID);
                    patt = mdict.getPattern(pattID);
                }
                if (layer == 0 && sta == 0)
                {
                    if (chipInMod == 4)
                        histClPositionCent[iMC]->Fill(npix);
                    else if (chipInMod == 8)
                        histClPositionForw[iMC]->Fill(npix);
                }
            }
        }
        if (iMC == 1)
            nPrimaries = 6.5e3;
        // histClPositionCent[iMC]->Scale(float(1. / nPrimaries));
        // histClPositionForw[iMC]->Scale(float(1. / nPrimaries));

        fPrimary->Close();
        fITSclus->Close();
    }
    gStyle->SetLegendTextSize(6);
    auto outFile = TFile("LHCC_chip_plots.root", "recreate");
    TCanvas cClusterSizeCent = TCanvas("cClusterSizeCent", "cClusterSizeCent");
    TCanvas cClusterSizeForw = TCanvas("cClusterSizeForw", "cClusterSizeForw");
    cClusterSizeCent.SetLogy();
    cClusterSizeForw.SetLogy();
    auto legend = new TLegend(0.33, 0.65, 0.95, 0.85);
    legend->SetMargin(0.1);

    cClusterSizeCent.cd();

    for (int iMC{0}; iMC < 2; iMC++)
    {

        auto color = iMC == 1 ? kRed + 2 : kBlue + 2;
        auto leg = iMC == 1 ? "MC" : "Data";

        histClPositionCent[iMC]->GetYaxis()->SetLabelSize(0.04);
        histClPositionCent[iMC]->GetYaxis()->SetTitleOffset(1.2);

        histClPositionCent[iMC]->GetYaxis()->CenterLabels();
        histClPositionCent[iMC]->SetStats(0);
        histClPositionCent[iMC]->SetLineWidth(2);

        histClPositionCent[iMC]->SetLineColor(color);
        histClPositionCent[iMC]->DrawNormalized("HIST SAME");
        legend->AddEntry(histClPositionCent[iMC], leg, "l");
        if (iMC == 1)
        {
            legend->SetHeader("ALICE pp #sqrt{s} = 900 GeV, ITS2 Layer 0, -0.64 < #eta < 0.64");
            legend->Draw();
        }
    }
    cClusterSizeCent.Write();

    cClusterSizeForw.cd();

    for (int iMC{0}; iMC < 2; iMC++)
    {
        auto color = iMC == 1 ? kRed + 2 : kBlue + 2;
        auto leg = iMC == 1 ? "MC" : "Data";

        histClPositionForw[iMC]->GetYaxis()->SetLabelSize(0.04);
        histClPositionForw[iMC]->GetYaxis()->SetTitleOffset(1.2);
        histClPositionForw[iMC]->GetYaxis()->CenterLabels();
        histClPositionForw[iMC]->SetStats(0);
        histClPositionForw[iMC]->SetLineWidth(2);
        histClPositionForw[iMC]->SetLineColor(color);
        histClPositionForw[iMC]->DrawNormalized("HIST SAME");
        histClPositionForw[iMC]->SetMaximum(1);

        if (iMC == 1)
        {
            legend->SetHeader("ALICE pp #sqrt{s} = 900 GeV, ITS2 Layer 0, 2.3 < #eta < 2.5");
            legend->Draw();
        }
    }
    cClusterSizeForw.Write();

    TCanvas cClusterSize = TCanvas("cClusterSizeCentForw", "cClusterSizeCentForw");
    cClusterSize.SetLogy();
    auto leg2 = new TLegend(0.33, 0.65, 0.95, 0.85);
    leg2->SetMargin(0.1);
    leg2->SetHeader("ALICE pp #sqrt{s} = 900 GeV, ITS2 Layer 0");
    histClPositionCent[0]->DrawNormalized("HIST SAME");
    histClPositionForw[0]->SetLineColor(kRed + 2);
    histClPositionForw[0]->SetLineWidth(2);
    histClPositionForw[0]->DrawNormalized("HIST SAME");
    leg2->AddEntry(histClPositionCent[0], "-0.64 < #eta < 0.64", "l");
    leg2->AddEntry(histClPositionForw[0], "2.3 < #eta < 2.5", "l");
    leg2->Draw();
    cClusterSize.Write();
    
}

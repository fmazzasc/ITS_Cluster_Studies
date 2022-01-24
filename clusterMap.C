#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
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

void fillIBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping);
void fillOBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping);

char *getIBLabel(int iBin, int layer);
char *getOBLabel(int iBin, int layer);

void clusterMap()
{
    gStyle->SetPalette(82);

    int pixelThr = 20;
    o2::itsmft::ChipMappingITS chipMapping;
    std::vector<TH2D *> ClSizeMaps(7);
    std::vector<TH1D *> AverageClSize(7);

    std::vector<int> nStaves{12, 16, 20, 24, 30, 42, 48};
    for (int layer{0}; layer < 7; layer++)
    {
        AverageClSize[layer] = new TH1D(Form("Average Cluster Size L%i", layer), "; Average cluster size; Counts", 99, 0.5, 99.5);

        if (layer < 3)
            ClSizeMaps[layer] = new TH2D(Form("chip map L%i", layer), Form("; Chip ID; Stave ID; # Clusters w/ size > %i / # Primary vertices", pixelThr), 9, -0.5, 8.5, nStaves[layer], -0.5, nStaves[layer] - 0.5);
        else
            ClSizeMaps[layer] = new TH2D(Form("chip map L%i", layer), Form("; Chip ID; Stave ID; # Clusters w/ size > %i / # Primary vertices", pixelThr), 49, -0.5, 48.5, 4 * nStaves[layer], -0.5, 4 * nStaves[layer] - 0.5);

        AverageClSize[layer]->SetLineStyle(5);
        AverageClSize[layer]->SetLineColor(kRed + 2);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    int nPrimaries = 0;

    std::vector<int> runNumbers = {505645, 505658};
    for (auto &runNum : runNumbers)
    {

        std::ostringstream strDir;
        strDir << runNum;
        auto dir = strDir.str();
        std::string o2clus_its_file = dir + "/" + "o2clus_its.root";
        std::string primary_vertex_file = dir + "/" + "o2_primary_vertex.root";

        auto fITSclus = TFile::Open(o2clus_its_file.data());
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        auto fPrimary = TFile::Open(primary_vertex_file.data());
        auto treePrimaries = (TTree *)fPrimary->Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;
        std::vector<o2::dataformats::PrimaryVertex> *Primaries = nullptr;

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
        treePrimaries->SetBranchAddress("PrimaryVertex", &Primaries);

        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {

            if (!treeITSclus->GetEvent(frame) || !treePrimaries->GetEvent(frame))
                continue;

            nPrimaries += Primaries->size();

            auto pattIt = ITSpatt->cbegin();

            for (auto &clus : *ITSclus)
            {
                auto pattID = clus.getPatternID();
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

                auto layer = gman->getLayer(clus.getSensorID());
                AverageClSize[layer]->Fill(npix);
                if (npix < pixelThr)
                    continue;
                layer < 3 ? fillIBmap(ClSizeMaps[layer], clus, chipMapping) : fillOBmap(ClSizeMaps[layer], clus, chipMapping);
            }
        }
        fPrimary->Close();
        fITSclus->Close();

        double scale = 1. / double(nPrimaries);

        TCanvas cClusterSize = TCanvas("cClusterSize", "cClusterSize", 1200, 800);
        cClusterSize.Divide(4, 2);

        auto outFile = TFile(Form("%i/clMaps_%i.root", runNum, pixelThr), "recreate");
        for (int layer{0}; layer < 7; layer++)
        {
            ClSizeMaps[layer]->Scale(scale);

            for (int i = 1; i < ClSizeMaps[layer]->GetNbinsY() + 1; i++)
            {
                if (layer < 3)
                {
                    if (i % 4 - 1 != 0)
                        ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, "");
                    else
                        ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, getIBLabel(i, layer));
                }
                else
                {
                    if (i % 8 - 1 != 0)
                        ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, "");
                    else
                        ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, getOBLabel(i, layer));
                    ;
                }
            }
            ClSizeMaps[layer]->GetYaxis()->SetLabelSize(0.04);
            ClSizeMaps[layer]->GetYaxis()->CenterLabels();
            ClSizeMaps[layer]->SetStats(0);
            ClSizeMaps[layer]->Write();

            auto c = cClusterSize.cd(layer + 1);
            c->SetLogy();
            AverageClSize[layer]->GetYaxis()->SetDecimals();
            AverageClSize[layer]->GetYaxis()->SetTitleOffset(1.2);
            AverageClSize[layer]->SetStats(1);
            AverageClSize[layer]->SetLineWidth(2);
            AverageClSize[layer]->DrawCopy();
        }

        cClusterSize.Write();
        outFile.Close();
    }
}

void fillIBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping)
{
    auto chipID = clus.getChipID();
    int lay, sta, ssta, mod, chipInMod;
    chipMapping.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);
    histo->Fill(chipInMod, sta);
}

void fillOBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping)
{
    auto chipID = clus.getChipID();
    int lay, sta, ssta, mod, chipInMod;
    chipMapping.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);
    auto xCoord = chipInMod < 7 ? (mod - 1) * 7 + chipInMod : (mod - 1) * 7 + 14 - chipInMod;
    auto yCoord = 4 * sta + ssta * 2 + 1 * (chipInMod < 7);
    histo->Fill(xCoord, yCoord);
}

char *getIBLabel(int iBin, int layer)
{
    return Form("L%i_%i", layer, iBin - 1);
}

char *getOBLabel(int iBin, int layer)
{
    return Form("L%i_%i", layer, (iBin - 1) / 4);
}
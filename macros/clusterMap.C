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

void fillIBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping, int weight = 1);
void fillOBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping, int weight = 1);

char *getIBLabel(int iBin, int layer);
char *getOBLabel(int iBin, int layer);

void clusterMap()
{
    gStyle->SetPalette(55);
    gStyle->SetPadRightMargin(0.25);
    //gStyle->SetPadLeftMargin(0.005);
    bool isMC = false;
    bool doClPositionNorm = false;
    bool doLHCCplots = true;
    const char *outFormat[2] = {".pdf", ".root"};

    std::vector<int> pixelThrs{20};
    std::vector<int> singleLayerClPosPlot{0, 1, 2};
    o2::itsmft::ChipMappingITS chipMapping;
    std::vector<TH2D *> ClSizeMaps(7);
    std::vector<TH1D *> AverageClSize(7);
    std::vector<TH2D *> AverageClSizeMap(7);
    std::vector<TH2D *> AverageOccupancyMap(7);
    std::vector<TH2D *> ClusterCounterMap(7);
    std::vector<TH2D *> histsClPosition(8);

    std::vector<int> nStaves{12, 16, 20, 24, 30, 42, 48};
    for (const auto& pixelThr: pixelThrs)
    {        
        for (int layer{0}; layer < 7; layer++)
        {
            AverageClSize[layer] = new TH1D(Form("Average Cluster Size L%i", layer), Form("; Cluster size for L%i; Counts/(# PVs)", layer), 99, 0.5, 99.5);

            if (layer < 3)
            {
                ClSizeMaps[layer] = new TH2D(Form("Big clusters map L%i", layer), Form("; Chip ID; Stave ID; (Cluster size > %i) / # PVs", pixelThr), 9, -0.5, 8.5, nStaves[layer], -0.5, nStaves[layer] - 0.5);
                AverageClSizeMap[layer] = new TH2D(Form("Cluster size map L%i", layer), "; Chip ID; Stave ID; #LT Cluster size #GT", 9, -0.5, 8.5, nStaves[layer], -0.5, nStaves[layer] - 0.5);
                AverageOccupancyMap[layer] = new TH2D(Form("Occupancy chip map L%i", layer), "; Chip ID; Stave ID; # Hits / # PVs", 9, -0.5, 8.5, nStaves[layer], -0.5, nStaves[layer] - 0.5);
                ClusterCounterMap[layer] = new TH2D(Form("Cluster counter map L%i", layer), "; Chip ID; Stave ID; # Clusters / # PVs", 9, -0.5, 8.5, nStaves[layer], -0.5, nStaves[layer] - 0.5);
            }

            else
            {
                ClSizeMaps[layer] = new TH2D(Form("Big clusters map L%i", layer), Form("; Chip ID; Stave ID; (Cluster size > %i) / # PVs", pixelThr), 49, -0.5, 48.5, 4 * nStaves[layer], -0.5, 4 * nStaves[layer] - 0.5);
                AverageClSizeMap[layer] = new TH2D(Form("Cluster size map L%i", layer), "; Chip ID; Stave ID; #LT Cluster size #GT", 49, -0.5, 48.5, 4 * nStaves[layer], -0.5, 4 * nStaves[layer] - 0.5);
                AverageOccupancyMap[layer] = new TH2D(Form("Occupancy chip map L%i", layer), "; Chip ID; Stave ID; #LT Cluster size #GT", 49, -0.5, 48.5, 4 * nStaves[layer], -0.5, 4 * nStaves[layer] - 0.5);
                ClusterCounterMap[layer] = new TH2D(Form("Cluster counter map L%i", layer), "; Chip ID; Stave ID; # Clusters / # PVs", 49, -0.5, 48.5, 4 * nStaves[layer], -0.5, 4 * nStaves[layer] - 0.5);
            }

            AverageClSize[layer]->SetLineStyle(5);
            AverageClSize[layer]->SetLineColor(kRed + 2);

            // CL position
            histsClPosition[layer] = new TH2D(Form("ClusterPositionVsL%i", layer), "; ; ; Counts", 1042, 0, 1042, 512, 0, 512);
        }

        // Geometry
        o2::base::GeometryManager::loadGeometry("utils/o2_geometry.root");
        auto gman = o2::its::GeometryTGeo::Instance();
        gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
        // Topology dictionary
        o2::itsmft::TopologyDictionary mdict;
        mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "utils/ITSdictionary.bin"));

        int nPrimaries = 0.;

        std::vector<int> runNumbers = {505645};
        for (auto &runNum : runNumbers)
        {
            std::ostringstream strDir;
            strDir << runNum;
            auto dir = strDir.str();
            strDir << runNum;
            std::string o2clus_its_file;
            std::string primary_vertex_file;
            if (!isMC)
            {
                o2clus_its_file = dir + "/" + "o2clus_its.root";
                primary_vertex_file = dir + "/" + "o2_primary_vertex.root";
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

            for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
            {

                if (!treeITSclus->GetEvent(frame) || !treePrimaries->GetEvent(frame))
                    continue;

                nPrimaries += Primaries->size();

                auto pattIt = ITSpatt->cbegin();

                for (auto &clus : *ITSclus)
                {
                    o2::itsmft::ClusterPattern patt;
                    auto layer = gman->getLayer(clus.getSensorID());
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

                    if (npix > pixelThr) // considering only "large" CL for CL position
                    {
                        auto col = clus.getCol();
                        auto row = clus.getRow();

                        // LOG(info) << "row: " << row << "col: " << col;
                        // LOG(info) << patt;

                        int ic = 0, ir = 0;

                        auto colSpan = patt.getColumnSpan();
                        auto rowSpan = patt.getRowSpan();
                        auto nBits = rowSpan * colSpan;

                        for (int i = 2; i < patt.getUsedBytes() + 2; i++)
                        {
                            unsigned char tempChar = patt.getByte(i);
                            int s = 128; // 0b10000000
                            while (s > 0)
                            {
                                if ((tempChar & s) != 0) // checking active pixels
                                {
                                    histsClPosition[layer]->Fill(col + ic, row + rowSpan - ir);
                                }
                                ic++;
                                s >>= 1;
                                if ((ir + 1) * ic == nBits)
                                {
                                    break;
                                }
                                if (ic == colSpan)
                                {
                                    ic = 0;
                                    ir++;
                                }
                                if ((ir + 1) * ic == nBits)
                                {
                                    break;
                                }
                            }
                        }
                    }
                    AverageClSize[layer]->Fill(npix);
                    layer < 3 ? fillIBmap(AverageClSizeMap[layer], clus, chipMapping, npix) : fillOBmap(AverageClSizeMap[layer], clus, chipMapping, npix);
                    layer < 3 ? fillIBmap(AverageOccupancyMap[layer], clus, chipMapping, npix) : fillOBmap(AverageOccupancyMap[layer], clus, chipMapping, npix);
                    layer < 3 ? fillIBmap(ClusterCounterMap[layer], clus, chipMapping) : fillOBmap(ClusterCounterMap[layer], clus, chipMapping);

                    if (npix < pixelThr)
                        continue;
                    layer < 3 ? fillIBmap(ClSizeMaps[layer], clus, chipMapping) : fillOBmap(ClSizeMaps[layer], clus, chipMapping);
                }
            }
            fPrimary->Close();
            fITSclus->Close();
            LOG(info) << nPrimaries;

            TCanvas cClusterSize = TCanvas("cClusterSize", "cClusterSize", 1200, 800);
            TCanvas cClusterPosition = TCanvas("cClusterPosition", "cClusterPosition", 1500, 1000);
            TCanvas cClusterPosition_L0 = TCanvas("cClusterPosition", "cClusterPosition", 1500, 1000);
            cClusterSize.Divide(4, 2);
            cClusterPosition.Divide(3, 3);

            // latex
            TLatex laClPos;
            laClPos.SetTextSize(0.06);
            laClPos.SetNDC();
            laClPos.SetTextFont(42);

            // LOG(info) << "Integral, " << AverageClSizeMap[0]->GetSumOfWeights() << ", entries: " << AverageClSizeMap[0]->GetEntries();

            auto outFile = TFile(Form("%i/clMaps_data_thr%i.root", runNum, pixelThr), "recreate");
            for (int layer{0}; layer < 7; layer++)
            {
                ClSizeMaps[layer]->Scale(1. / nPrimaries);
                AverageOccupancyMap[layer]->Scale(1. / nPrimaries);
                AverageClSizeMap[layer]->Divide(ClusterCounterMap[layer]);
                ClusterCounterMap[layer]->Scale(1. / nPrimaries);
                if (doClPositionNorm)
                {
                    histsClPosition[layer]->Scale(1. / nPrimaries);
                }

                for (int i = 1; i < ClSizeMaps[layer]->GetNbinsY() + 1; i++)
                {
                    if (layer < 3)
                    {
                        if (i % 4 - 1 != 0)
                        {
                            ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, "");
                            AverageClSizeMap[layer]->GetYaxis()->SetBinLabel(i, "");
                            AverageOccupancyMap[layer]->GetYaxis()->SetBinLabel(i, "");
                        }

                        else
                        {
                            ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, getIBLabel(i, layer));
                            AverageClSizeMap[layer]->GetYaxis()->SetBinLabel(i, getIBLabel(i, layer));
                            AverageOccupancyMap[layer]->GetYaxis()->SetBinLabel(i, getIBLabel(i, layer));
                        }
                    }
                    else
                    {
                        if (i % 8 - 1 != 0)
                        {
                            ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, "");
                            AverageClSizeMap[layer]->GetYaxis()->SetBinLabel(i, "");
                        }
                        else
                        {
                            ClSizeMaps[layer]->GetYaxis()->SetBinLabel(i, getOBLabel(i, layer));
                            AverageClSizeMap[layer]->GetYaxis()->SetBinLabel(i, getOBLabel(i, layer));
                        }
                    }
                }
                ClSizeMaps[layer]->GetYaxis()->SetLabelSize(0.04);
                ClSizeMaps[layer]->GetYaxis()->CenterLabels();
                ClSizeMaps[layer]->GetZaxis()->SetTitleOffset(0.9);
                ClSizeMaps[layer]->SetStats(0);
                ClSizeMaps[layer]->Write();

                AverageClSizeMap[layer]->GetYaxis()->SetLabelSize(0.04);
                AverageClSizeMap[layer]->GetYaxis()->CenterLabels();
                AverageClSizeMap[layer]->GetZaxis()->SetTitleOffset(0.9);
                AverageClSizeMap[layer]->SetStats(0);
                if (doLHCCplots && (layer == 0))
                {
                    TCanvas cAverClusPosLhcc = TCanvas("cAvClusSizeMapLhcc", "cAvClusSizeMapLhcc", 1400, 1200);
                    cAverClusPosLhcc.cd();
                    AverageClSizeMap[layer]->GetYaxis()->SetLabelSize(0.045);
                    AverageClSizeMap[layer]->GetZaxis()->SetLabelOffset(0.005);
                    AverageClSizeMap[layer]->GetZaxis()->SetTitleOffset(1.2);
                    const char* AvClSizeMapLhccTitle = (isMC) ? Form("ALICE - MC pp #sqrt{s} = 900 GeV, run %i", runNum) : Form("ALICE - pp #sqrt{s} = 900 GeV, run %i", runNum);
                    AverageClSizeMap[layer]->SetTitle(AvClSizeMapLhccTitle);
                    AverageClSizeMap[layer]->Draw("colz");
                    for (int i = 0; i < 2; i++)
                    {
                        const char* ClusMapSizeLhccTitle = (isMC) ? Form("cAvClusSizeMapLhcc_L0MC%s", outFormat[i]) : Form("cAvClusSizeMapLhcc_L0data%s", outFormat[i]);
                        cAverClusPosLhcc.SaveAs(Form("LHCCplots/%s", ClusMapSizeLhccTitle));
                    }
                }
                AverageClSizeMap[layer]->Write();

                AverageOccupancyMap[layer]->GetYaxis()->SetLabelSize(0.04);
                AverageOccupancyMap[layer]->GetYaxis()->CenterLabels();
                AverageOccupancyMap[layer]->GetZaxis()->SetTitleOffset(0.9);
                AverageOccupancyMap[layer]->SetStats(0);
                AverageOccupancyMap[layer]->Write();

                ClusterCounterMap[layer]->GetYaxis()->SetLabelSize(0.04);
                ClusterCounterMap[layer]->GetYaxis()->CenterLabels();
                ClusterCounterMap[layer]->GetZaxis()->SetTitleOffset(0.9);
                ClusterCounterMap[layer]->SetStats(0);
                ClusterCounterMap[layer]->Write();

                TH2D *ClusTimesOcc = (TH2D *)AverageClSizeMap[layer]->Clone(Form("Cluster size x Occupancy chip map L%i", layer));
                ClusTimesOcc->Multiply(AverageOccupancyMap[layer]);
                ClusTimesOcc->GetZaxis()->SetTitle("#LT Cluster size #GT x < Occupancy >");
                ClusTimesOcc->Write();

                auto c = cClusterSize.cd(layer + 1);
                c->SetLogy();
                AverageClSize[layer]->GetYaxis()->SetDecimals();
                AverageClSize[layer]->GetYaxis()->SetTitleOffset(1.);
                AverageClSize[layer]->SetStats(1);
                AverageClSize[layer]->SetLineWidth(2);
                AverageClSize[layer]->DrawCopy();

                cClusterPosition.cd(layer + 1);
                histsClPosition[layer]->SetTitle(Form("L%i CL Position", layer));
                histsClPosition[layer]->Draw("colz0");
                if (layer == singleLayerClPosPlot.at(layer))
                {
                    cClusterPosition_L0.cd();
                    histsClPosition[layer]->GetZaxis()->SetTitle(Form("CL > %i", pixelThr));
                    histsClPosition[layer]->Draw("colz0");
                    for (int i = 0; i < 2; i++)
                    {
                        cClusterPosition_L0.SaveAs(Form("cClPosition_L%i_100%s", singleLayerClPosPlot.at(layer), outFormat[i]));
                    }
                }
                if (layer == 0) // cloning CL position on layer 0 into overall CL position
                {
                    histsClPosition[8] = new TH2D();
                    histsClPosition[8]->Clone(Form("ClusterPositionVsL%i", layer));
                    histsClPosition[8]->SetTitle("Overall CL Position");
                }
                else if (layer + 2 == 8)
                {
                    cClusterPosition.cd(layer + 2);
                    if (doClPositionNorm)
                    {
                        histsClPosition[8]->Scale(1. / nPrimaries);
                    }
                    histsClPosition[8]->Draw("colz");
                    cClusterPosition.cd(layer + 3);
                    laClPos.DrawLatex(0.15, 0.6, Form("Large cluster position in ITS chip (Cl size > %i)", pixelThr));
                }
                histsClPosition[8]->Add(histsClPosition[layer]); // Adding layers
            }
            cClusterPosition.Write();
            for (int i = 0; i < 2; i++)
            {
                const char* strClusPos = (isMC) ? Form("cClPosition_MC_thr%i%s", pixelThr, outFormat[i]) : Form("cClPosition_thr%i%s", pixelThr, outFormat[i]);
                cClusterPosition.SaveAs(strClusPos);
            } 
            AverageClSize[0]->GetYaxis()->SetTitleOffset(1.);
            cClusterSize.Write();
            outFile.Close();
        }
    }
}

void fillIBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping, int weight)
{
    auto chipID = clus.getChipID();
    int lay, sta, ssta, mod, chipInMod;
    chipMapping.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);
    histo->Fill(chipInMod, sta, weight);
}

void fillOBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping, int weight)
{
    auto chipID = clus.getChipID();
    int lay, sta, ssta, mod, chipInMod;
    chipMapping.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);
    auto xCoord = chipInMod < 7 ? (mod - 1) * 7 + chipInMod : (mod - 1) * 7 + 14 - chipInMod;
    auto yCoord = 4 * sta + ssta * 2 + 1 * (chipInMod < 7);
    histo->Fill(xCoord, yCoord, weight);
}

char *getIBLabel(int iBin, int layer)
{
    return Form("L%i_%i", layer, iBin - 1);
}

char *getOBLabel(int iBin, int layer)
{
    return Form("L%i_%i", layer, (iBin - 1) / 4);
}
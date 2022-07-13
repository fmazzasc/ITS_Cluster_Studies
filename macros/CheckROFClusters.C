#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include <TRandom.h>
#include <gsl/gsl>
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2D.h"
#include "TH3I.h"
#include "TH1D.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"

#endif

using namespace o2::itsmft;
using CompClusterExt = o2::itsmft::CompClusterExt;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

void checkROFClusters()
{
    gStyle->SetPalette(55);
    gStyle->SetPadRightMargin(0.35);
    // gStyle->SetPadLeftMargin(0.005);
    bool isMC = false;
    bool verbose = false;
    int pixelThr{40};
    int selectedFrame = 730;
    int initialRof = 487;
    int finalRof = 491;

    // Geometry
    o2::base::GeometryManager::loadGeometry("utils/o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "utils/ITSdictionary.bin"));

    std::vector<int> rofClusters;
    TH3I *hRofClFeatures = new TH3I("cl feat", "", 1024, -0.5, 1023.5, 512, -0.5, 511.5, 1024, -0.5, 1023.5); // evaluate clusters repetition
    TH2I *hRofHisto = new TH2I("ROF info", "", 1024, -0.5, 1023.5, 512, -0.5, 511.5);                         // evaluate clusters repetition
    TH2I *hTFHisto = new TH2I("TF info", "", 1024, -0.5, 1023.5, 512, -0.5, 511.5);                           // evaluate clusters repetition
    auto repeatCounter = 0;

    std::vector<int> runNumbers = {505645};
    std::ofstream chekROFtxt;
    for (auto &runNum : runNumbers)
    {
        chekROFtxt.open (Form("checkROFCluster_run%i.txt", runNum));
        chekROFtxt << "col" << " " << "row" << " " << "npixels" << " " << "rofInd+1" << " " << "TF\n";
        std::ostringstream strDir;
        strDir << runNum;
        auto dir = strDir.str();
        strDir << runNum;
        std::string o2clus_its_file;
        std::string primary_vertex_file;
        if (!isMC)
        {
            o2clus_its_file = dir + "/" + "o2clus_its.root";
        }
        else
        {
            o2clus_its_file = "ITS_MC/o2clus_its.root";
        }

        auto fITSclus = TFile::Open(o2clus_its_file.data());
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<o2::itsmft::ROFRecord> *ITSrof = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClustersROF", &ITSrof);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {
            hRofHisto->Reset();
            if (!treeITSclus->GetEvent(frame))
                continue;

            std::vector<TH2D *> hHitMapsVsFrame((finalRof - initialRof));
            auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());

            if (frame == selectedFrame)
            {     
                for (int i = 0;  i < (finalRof - initialRof); i++)
                {
                    hHitMapsVsFrame[i] = new TH2D(Form("hHitMapsVsFrame_TF%i_rof%i", frame, initialRof+i), "; ; ; Counts", 1024, 0, 1024, 512, 0, 512);
                }
            } 

            for (unsigned int rofInd{0}; rofInd < ITSrof->size(); rofInd++)
            {
                const auto &rof = (*ITSrof)[rofInd];
                int rof_counter = 0;
                auto clustersInFrame = rof.getROFData(*ITSclus);
                auto pattIt = ITSpatt->cbegin();

                for (unsigned int clusInd{0}; clusInd < clustersInFrame.size(); clusInd++)
                {
                    const auto &clus = clustersInFrame[clusInd];
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
                        rof_counter++;
                        if (frame == selectedFrame)
                        {
                            if ( initialRof < int(rofInd) && int(rofInd) < finalRof)
                            {
                                auto col = clus.getCol();
                                auto row = clus.getRow();
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
                                            hHitMapsVsFrame[rofInd - initialRof]->Fill(col + ic, row + rowSpan - ir);
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
                        }
                        chekROFtxt << clus.getCol() << " " << clus.getRow() << " " << npix << " " << rofInd + 1 << " " << frame << "\n";
                        if (hRofClFeatures->GetBinContent(clus.getCol(), clus.getRow(), npix) == 0)
                        {
                            hRofClFeatures->Fill(clus.getCol(), clus.getRow(), npix);
                            hRofHisto->Fill(clus.getCol(), clus.getRow(), rofInd + 1);
                            hTFHisto->Fill(clus.getCol(), clus.getRow(), frame);
                        }
                        else
                        {
                            if (hRofHisto->GetBinContent(clus.getCol(), clus.getRow()) != rofInd)
                            {
                                if (verbose)
                                {
                                    LOG(info) << "----------------------------------------------------------";
                                    LOG(info) << "RANDOM Repetition found for Cluster with row: " << clus.getRow() << ", col: " << clus.getCol() << ", Npix: " << npix;
                                    LOG(info) << "Cluster ROF" << rofInd << ", Cluster TF:" << frame << ", Filled ROF: " << hRofHisto->GetBinContent(clus.getCol(), clus.getRow()) << ", Filled TF: " << hTFHisto->GetBinContent(clus.getCol(), clus.getRow());
                                }
                                repeatCounter++;
                                hRofClFeatures->Fill(clus.getCol(), clus.getRow(), npix);
                                hRofHisto->Fill(clus.getCol(), clus.getRow(), rofInd+1);
                                hTFHisto->Fill(clus.getCol(), clus.getRow(), frame);
                            }
                            else
                            {
                                if (verbose)
                                {
                                    LOG(info) << "Repetition found for Cluster with row: " << clus.getRow() << ", col: " << clus.getCol() << ", Npix: " << npix << ", ROF: " << rofInd << ", TF:" << frame;
                                }
                            }
                        }
                    }
                    rofClusters.push_back(rof_counter);
                }
            }

            if (frame == selectedFrame)
            {
                auto HitmapFile = TFile("HitmapRof.root", "recreate");
                TCanvas cHitmapRof = TCanvas("cHitmapRof", "cHitmapRof", 1200, 800);
                cHitmapRof.Divide(1, 2);
                for (int rofInd{0}; rofInd < (finalRof - initialRof); rofInd++)
                {
                    cHitmapRof.cd(rofInd+1);
                    hHitMapsVsFrame[rofInd]->Draw("colz0");
                    hHitMapsVsFrame[rofInd]->Write();
                }
                cHitmapRof.SaveAs("cHitmapRof.pdf");
                cHitmapRof.Write();
                HitmapFile.Close();
            }
        }
    }
    chekROFtxt.close();

    TH1D *clRofHisto = new TH1D("rof histo", ";ROF num; Counts", rofClusters.size(), 0, rofClusters.size() - 1);
    for (unsigned int iBin{1}; iBin <= rofClusters.size(); iBin++)
    {
        if (rofClusters[iBin - 1] > 0)
            clRofHisto->SetBinContent(iBin, rofClusters[iBin - 1]);
    }
    if(verbose)
    {
        LOG(info) << "# Clusters: " << clRofHisto->Integral();
        LOG(info) << "# Unique Clusters : " << hRofClFeatures->GetEntries();
        LOG(info) << "Random repetition : " << repeatCounter;
    }

    auto file = TFile("rofclus.root", "recreate");
    clRofHisto->Write();
    file.Close();
}
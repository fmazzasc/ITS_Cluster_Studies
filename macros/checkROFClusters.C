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
    gStyle->SetPalette(82);
    gStyle->SetPadRightMargin(0.35);
    // gStyle->SetPadLeftMargin(0.005);
    bool isMC = false;
    int pixelThr{0};

    // Geometry
    o2::base::GeometryManager::loadGeometry("utils/o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "utils/ITSdictionary.bin"));

    std::vector<int> rofClusters;
    TH3I *hRofClFeatures = new TH3I("cl feat", "", 1024, -0.5, 1023.5, 512, -0.5, 511 / 5, 1024, -0.5, 1023.5); // evaluate clusters repetition
    TH2I *hRofHisto = new TH2I("ROF info", "", 1024, -0.5, 1023.5, 512, -0.5, 511 / 5);                         // evaluate clusters repetition
    TH2I *hTFHisto = new TH2I("TF info", "", 1024, -0.5, 1023.5, 512, -0.5, 511 / 5);                           // evaluate clusters repetition
    auto repeatCounter = 0;

    std::vector<int> runNumbers = {505645};
    std::ofstream myfile;
    for (auto &runNum : runNumbers)
    {
        myfile.open (Form("checkROFCluster_run%i.txt", runNum));
        myfile << "clus.getCol()" << " " << "clus.getRow()" << " " << "Npixels" << " " << "rofInd+1" << " " << "TF\n";
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

            auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());

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
                        myfile << clus.getCol() << " " << clus.getRow() << " " << npix << " " << rofInd + 1 << " " << frame << "\n";
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
                                //LOG(info) << "----------------------------------------------------------";
                                //LOG(info) << "RANDOM Repetition found for Cluster with row: " << clus.getRow() << ", col: " << clus.getCol() << ", Npix: " << npix;
                                //LOG(info) << "Cluster ROF" << rofInd << ", Cluster TF:" << frame << ", Filled ROF: " << hRofHisto->GetBinContent(clus.getCol(), clus.getRow()) << ", Filled TF: " << hTFHisto->GetBinContent(clus.getCol(), clus.getRow());
                                repeatCounter++;
                                hRofClFeatures->Fill(clus.getCol(), clus.getRow(), npix);
                                hRofHisto->Fill(clus.getCol(), clus.getRow(), rofInd+1);
                                hTFHisto->Fill(clus.getCol(), clus.getRow(), frame);
                            }
                            else
                            {
                                //LOG(info) << "Repetition found for Cluster with row: " << clus.getRow() << ", col: " << clus.getCol() << ", Npix: " << npix << ", ROF: " << rofInd << ", TF:" << frame;
                            }
                        }
                    }
                }
                rofClusters.push_back(rof_counter);
            }
        }
    }
    myfile.close();

    TH1D *clRofHisto = new TH1D("rof histo", ";ROF num; Counts", rofClusters.size(), 0, rofClusters.size() - 1);
    for (unsigned int iBin{1}; iBin <= rofClusters.size(); iBin++)
    {
        if (rofClusters[iBin - 1] > 0)
            clRofHisto->SetBinContent(iBin, rofClusters[iBin - 1]);
    }
    //LOG(info) << "# Clusters: " << clRofHisto->Integral();
    //LOG(info) << "# Unique Clusters : " << hRofClFeatures->GetEntries();
    //LOG(info) << "Random repetition : " << repeatCounter;

    auto file = TFile("rofclus.root", "recreate");
    clRofHisto->Write();
    file.Close();
}
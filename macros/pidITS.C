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

void pidITS()
{

    TH2D *hSplines = new TH2D("ITS splines ", ";#it{p}^{ITS-TPC}; < Cluster size > ; Counts", 300, 0, 2, 80, 0.5, 16.5);

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    std::vector<int> runNumbers = {505658};

    for (auto &runNum : runNumbers)
    {
        std::ostringstream strDir;
        strDir << runNum;
        auto dir = strDir.str();

        std::string o2match_itstpc_file = dir + "/" + "o2match_itstpc.root";

        std::string o2trac_its_file = dir + "/" + "o2trac_its.root";
        std::string o2clus_its_file = dir + "/" + "o2clus_its.root";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

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

        std::vector<float> pakiWeights{0.11, 0.7, 0.7, 0.75, 0.7, 0.6, 0.09};
        float sumPaki = std::accumulate(pakiWeights.begin(), pakiWeights.end(), 0.);

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {
            LOG(info) << frame;

            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame))
                continue;

            auto pattIt = ITSpatt->cbegin();

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {
                // if(frame>=80575){
                // break;
                // // LOG(info) << TPCITStracks->size();
                // // LOG(info) << ITStracks->size();
                // // LOG(info) << ITSclus->size();

                
                // }


                auto &ITSTPCtrack = TPCITStracks->at(iTrack);

                if(ITSTPCtrack.getRefITS()>ITStracks->size()-1)
                break;

                auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());

                std::vector<CompClusterExt> TrackClus;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus.push_back(clus);
                }

                std::reverse(TrackClus.begin(), TrackClus.end());
                std::vector<int> clusterSizes;

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
                        clusterSizes.push_back(npix);
                    }
                }
                std::sort(clusterSizes.begin(), clusterSizes.end());
                float mean = 0;
                for (unsigned int i{0}; i < clusterSizes.size(); i++)
                {
                    mean += clusterSizes[i]*pakiWeights[6-i];
                }
                mean /= sumPaki;
                if(clusterSizes.size()==7 && std::abs(ITSTPCtrack.getEta()) < 0.5)
                    hSplines->Fill(ITSTPCtrack.getP(), mean/std::abs(ITSTPCtrack.getTgl()));
            }
        }
        treeITS->ResetBranchAddresses();
        treeITSTPC->ResetBranchAddresses();
        treeITSclus->ResetBranchAddresses();
        fITS->Close();
        fITSTPC->Close();
        fITSclus->Close();
    }
    TFile outFile = TFile("pid.root", "recreate"); 
    hSplines->Write();
    outFile.Close();
}

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

void checkClusters()
{
    std::vector<TH2D *> histsPt(7);
    std::vector<TH2D *> histsEta(7);

    for (int layer{0}; layer < 7; layer++)
    {
        std::ostringstream str;
        str << "Cluster Size for L" << layer;
        std::string histsName = str.str();
        histsPt[layer] = new TH2D((histsName + "vs pT").data(), ("; " + histsName + "; #it{p}_{T}^{ITS-TPC} (GeV/#it{c}); Counts").data(), 14, 1, 15, 30, 0, 5);
        histsEta[layer] = new TH2D((histsName + "vs eta").data(), ("; " + histsName + "; #eta; Counts").data(), 14, 1, 15, 40, -2, 2);
    }

    TH1D *hPtRes = new TH1D("pT resolution ", ";(#it{p}_{T}^{ITS} - #it{p}_{T}^{ITS-TPC})/#it{p}_{T}^{ITS-TPC}; Counts", 80, -1, 1);

    std::vector<int> runNumbers = {505658};
    for (auto &runNum : runNumbers)
    {
        std::ostringstream strDir;
        strDir << runNum;
        auto dir = strDir.str();

        std::string o2match_itstpc_file = dir + "/" + "o2match_itstpc.root";

        std::string o2trac_its_file = dir + "/" + "o2trac_its.root";
        std::string o2clus_its_file = dir + "/" + "o2clus_its.root";

        std::string geomFile = "o2";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

        // Geometry
        o2::base::GeometryManager::loadGeometry(geomFile.data());
        auto gman = o2::its::GeometryTGeo::Instance();
        gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

        auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
        auto treeITS = (TTree *)fITS->Get("o2sim");
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        // Topology dictionary
        o2::itsmft::TopologyDictionary mdict;
        mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

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

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {
            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame))
                continue;

            auto pattIt = ITSpatt->cbegin();

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {
                auto &ITSTPCtrack = TPCITStracks->at(iTrack);
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

                hPtRes->Fill((ITStrack.getPt() - ITSTPCtrack.getPt()) / ITSTPCtrack.getPt());

                std::reverse(TrackClus.begin(), TrackClus.end());

                for (int layer{0}; layer < 6; layer++)
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

                        histsPt[layer]->Fill(npix, ITSTPCtrack.getPt());
                        histsEta[layer]->Fill(npix, ITSTPCtrack.getEta());
                    }
                }
            }
        }
    }

    auto outFile = TFile("clusITS.root", "recreate");

    for (int layer{0}; layer < 7; layer++)
    {
        histsPt[layer]->Write();
        histsEta[layer]->Write();
    }
    hPtRes->Write();
}
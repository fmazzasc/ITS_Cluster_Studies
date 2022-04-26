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

void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern> &pattVec, std::vector<CompClusterExt> *ITSclus, std::vector<unsigned char> *ITSpatt, o2::itsmft::TopologyDictionary &mdict, o2::its::GeometryTGeo *gman);
void printClusTrackInfo(const std::array<CompClusterExt, 7> &TrackClus, const std::array<o2::itsmft::ClusterPattern, 7> &TrackPatt, o2::its::TrackITS &ITStrack);
void fillClusterMap(CompClusterExt &clus, o2::itsmft::ClusterPattern &patt, TH2D *histo);

void bigClusterAnalyser()
{
    bool useITSonly = true;
    std::string itsOnlyStr = useITSonly ? "ITS-SA" : "ITS-TPC";

    double ptmax = 5;
    double ptbins = ptmax / 0.033;

    int clsize_min = 50;
    int clsize_max = 150;
    std::vector<TH2D *> histsClMapTracks(7);
    std::vector<TH2D *> histsClMapNoTracks(7);

    for (int layer{0}; layer < 7; layer++)
    {
        histsClMapTracks[layer] = new TH2D(Form("ClMapTrackL%i", layer), "; Column; Row ; Hits", 1024, -0.5, 1023.5, 512, -0.5, 511.5);
        histsClMapNoTracks[layer] = new TH2D(Form("ClMapNoTrackL%i", layer), "; Column; Row ; Hits", 1024, -0.5, 1023.5, 512, -0.5, 511.5);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    std::string path = "/data/fmazzasc/its_data/505658";
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
        if (counter > 100)
            continue;
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

            std::vector<int> clusTrackIdxs;
            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);


            int trackSize = useITSonly ? ITStracks->size() : TPCITStracks->size();

            for (int iTrack{0}; iTrack < trackSize; ++iTrack)
            {

                int trackPos = useITSonly ? iTrack : int(TPCITStracks->at(iTrack).getRefITS());
                auto &ITStrack = ITStracks->at(trackPos);

                o2::track::TrackParCov baseTrack(useITSonly ? (o2::track::TrackParCov)ITStrack : TPCITStracks->at(iTrack));

                std::array<CompClusterExt, 7> TrackClus;
                std::array<o2::itsmft::ClusterPattern, 7> TrackPatt;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &patt = pattVec[(*ITSTrackClusIdx)[firstClus + icl]];
                    // LOG(info) << "Patt Npixels: " << pattVec[0].getNPixels();

                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus[layer] = clus;
                    TrackPatt[layer] = patt;
                    clusTrackIdxs.push_back((*ITSTrackClusIdx)[firstClus + icl]);
                }

                for (int layer{0}; layer < 7; layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {

                        auto &pattern = TrackPatt[layer];
                        auto npix = pattern.getNPixels();

                        if (npix > 50)
                        {
                            // LOG(info) << "------------------------------------------------------";
                            // LOG(info) << "Track " << iTrack << " has cluster on layer " << layer << " and " << npix << " pixels";
                            // printClusTrackInfo(TrackClus, TrackPatt, ITStrack);
                            fillClusterMap(TrackClus[layer], pattern, histsClMapTracks[layer]);
                        }
                    }
                }
            }
            for (unsigned int iClus{0}; iClus < ITSclus->size(); ++iClus)
            {

                bool isTrackClus = false;
                for (auto idx : clusTrackIdxs)
                {
                    if (idx == int(iClus))
                    {
                        isTrackClus = true;
                        break;
                    }
                }
                if (isTrackClus)
                    continue;
                auto &clus = (*ITSclus)[iClus];
                auto &pattern = pattVec[iClus];
                auto npix = pattern.getNPixels();

                auto layer = gman->getLayer(clus.getSensorID());
                if (npix > 50)
                {
                    fillClusterMap(clus, pattern, histsClMapNoTracks[layer]);
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

    auto outFile = TFile("cluster_map.root", "recreate");
    for (int iLayer{0}; iLayer < 7; iLayer++)
    {
        histsClMapTracks[iLayer]->Write();
        histsClMapNoTracks[iLayer]->Write();
    }
    outFile.Close();
}

void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern> &pattVec, std::vector<CompClusterExt> *ITSclus, std::vector<unsigned char> *ITSpatt, o2::itsmft::TopologyDictionary &mdict, o2::its::GeometryTGeo *gman)
{
    pattVec.reserve(ITSclus->size());
    auto pattIt = ITSpatt->cbegin();
    for (unsigned int iClus{0}; iClus < ITSclus->size(); ++iClus)
    {
        auto &clus = (*ITSclus)[iClus];
        auto layer = gman->getLayer(clus.getSensorID());

        auto pattID = clus.getPatternID();
        int npix;
        o2::itsmft::ClusterPattern patt;

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
        // LOG(info) << "npix: " << npix << " Patt Npixel: " << patt.getNPixels();
        pattVec.push_back(patt);
    }
        // LOG(info) << " Patt Npixel: " << pattVec[0].getNPixels();
}
void printClusTrackInfo(const std::array<CompClusterExt, 7> &TrackClus, const std::array<o2::itsmft::ClusterPattern, 7> &TrackPatt, o2::its::TrackITS &ITStrack)
{
    for (int layer{0}; layer < 7; layer++)
    {
        if (ITStrack.hasHitOnLayer(layer))
        {

            LOG(info) << "Layer " << layer << ": " << TrackPatt[layer].getNPixels() << " pixels";
            LOG(info) << "Track P: " << ITStrack.getP() << " Eta: " << ITStrack.getEta();
        }
    }
}
void fillClusterMap(CompClusterExt &clus, o2::itsmft::ClusterPattern &patt, TH2D *histo)
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
                histo->Fill(col + ic, row + rowSpan - ir);
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
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
void TrackLayerCorr(o2::its::TrackITS &ITStrack, const std::array<CompClusterExt, 7>clus, const std::array<o2::itsmft::ClusterPattern, 7>patt, TH1D* hClSizeAll, std::vector<TH1D *> hClSizeAllvsLayer, int pixThr, std::string cut);
static bool npix_compare(o2::itsmft::ClusterPattern a, o2::itsmft::ClusterPattern b);

void bigClusterAnalyser()
{
    bool useITSonly = true;
    bool doTrckClusCorr = false;
    bool doCluShape = true;

    std::string itsOnlyStr = useITSonly ? "ITS-SA" : "ITS-TPC";

    double ptmax = 5;
    double ptbins = ptmax / 0.033;

    int pixThr = 50;
    int clsize_min = 50;
    int clsize_max = 150;
    std::vector<TH2D *> histsClMapTracks(7);
    std::vector<TH2D *> histsClMapNoTracks(7);

    // cluster correlation
    TH1D *hClSizeCorrAllLow = new TH1D("hClSizeCorrAllLow", ";Cluster size;Entries", 50, 0.5, 50.5);
    std::vector<TH1D *> hClSizeCorrVsLayerLow(7);
    TH1D *hClSizeCorrAllHigh = new TH1D("hClSizeCorrAllHigh", ";Cluster size;Entries", 50, 0.5, 50.5);
    std::vector<TH1D *> hClSizeCorrVsLayerHigh(7);

    // cluster shape
    std::vector<TH1D *> hClSigmaXvsLayer(7);
    std::vector<TH1D *> hClSigmaYvsLayer(7);

    for (int layer{0}; layer < 7; layer++)
    {
        histsClMapTracks[layer] = new TH2D(Form("ClMapTrackL%i", layer), "; Column; Row ; Hits", 1024, -0.5, 1023.5, 512, -0.5, 511.5);
        histsClMapNoTracks[layer] = new TH2D(Form("ClMapNoTrackL%i", layer), "; Column; Row ; Hits", 1024, -0.5, 1023.5, 512, -0.5, 511.5);

        if (doTrckClusCorr)
        {
            hClSizeCorrVsLayerLow[layer] = new TH1D(Form("hClSizeCorrVsLayerLowL%i", layer), "; Cluster size; Entries", 50, 0.5, 50.5);
            hClSizeCorrVsLayerHigh[layer] = new TH1D(Form("hClSizeCorrVsLayerHighL%i", layer), "; Cluster size; Entries", 50, 0.5, 50.5);
        }

        if(doCluShape)
        {
            hClSigmaXvsLayer[layer] = new TH1D(Form("hClSigmaXvsLayerL%i", layer), "; Cluster size; Entries", 50, 0.5, 50.5);
            hClSigmaYvsLayer[layer] = new TH1D(Form("hClSigmaYvsLayerL%i", layer), "; Cluster size; Entries", 50, 0.5, 50.5);
        }
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("../utils/o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "../utils/ITSdictionary.bin"));

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
        if (counter > 10)
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

                if (doTrckClusCorr)
                {
                    TrackLayerCorr(ITStrack, TrackClus, TrackPatt, hClSizeCorrAllHigh, hClSizeCorrVsLayerHigh, pixThr, "upper"); 
                    TrackLayerCorr(ITStrack, TrackClus, TrackPatt, hClSizeCorrAllLow, hClSizeCorrVsLayerLow, pixThr, "lower");
                }

                for (int layer{0}; layer < 7; layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {

                        auto &pattern = TrackPatt[layer];
                        auto npix= pattern.getNPixels();

                        if (npix > 50)
                        {
                            // LOG(info) << "------------------------------------------------------";
                            // LOG(info) << "Track " << iTrack << " has cluster on layer " << layer << " and " << npix << " pixels";
                            // printClusTrackInfo(TrackClus, TrackPatt, ITStrack);
                            fillClusterMap(TrackClus[layer], pattern, histsClMapTracks[layer]);
                            if (doCluShape)
                            {
                                hClSigmaXvsLayer[layer]->Fill(pattern.getRowSpan());
                                hClSigmaYvsLayer[layer]->Fill(pattern.getColumnSpan());
                            }
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

    // Saving correlation plots
    if (doTrckClusCorr)
    {
        auto outFileCorr = TFile(Form("clusITStrackCorr%iUpdate.root", pixThr), "recreate");
        TCanvas clusITStrackCorr = TCanvas("clusITStrackCorr", "clusITStrackCorr", 800, 800);
        clusITStrackCorr.cd()->SetLogy();
        hClSizeCorrAllHigh->SetLineColor(kRed);
        hClSizeCorrAllHigh->DrawNormalized();
        hClSizeCorrAllLow->DrawNormalized("same");
        clusITStrackCorr.Write();
        hClSizeCorrAllHigh->Write();
        hClSizeCorrAllLow->Write();

        for (int layer{0}; layer < 7; layer++)
        {
            TCanvas cClusterSize = TCanvas(Form("cClusterSizeL%i", layer), Form("cClusterSizeL%i", layer));
            cClusterSize.SetLogy();
            auto leg2 = new TLegend(0.33, 0.65, 0.8, 0.85);
            leg2->SetBorderSize(0);
            leg2->SetHeader(Form("ALICE pp #sqrt{s} = 900 GeV, ITS2 Layer %i - thr = %i", layer, pixThr));
            leg2->AddEntry(hClSizeCorrVsLayerHigh[layer], "Correlated clusters", "l");
            leg2->AddEntry(hClSizeCorrVsLayerLow[layer], "All clusters", "l");
        
            hClSizeCorrVsLayerHigh[layer]->SetLineColor(kRed);
            hClSizeCorrVsLayerHigh[layer]->DrawNormalized("hist");
            hClSizeCorrVsLayerLow[layer]->DrawNormalized("hist same");
            leg2->Draw();

            cClusterSize.Write();
            hClSizeCorrVsLayerHigh[layer]->Write();
            hClSizeCorrVsLayerLow[layer]->Write();
        }
        outFileCorr.Close();
    }

    // Saving shape plots
    if (doCluShape)
    {
        auto outFileShape = TFile(Form("outFileShape_Thr%i.root", pixThr), "recreate");

        for (int layer{0}; layer < 7; layer++)
        {
            TCanvas cClusterSize = TCanvas(Form("cClusterSizeL%i", layer), Form("cClusterSizeL%i", layer));
            cClusterSize.SetLogy();
            hClSigmaXvsLayer[layer]->SetLineColor(kRed);
            hClSigmaYvsLayer[layer]->SetLineColor(kBlue);
            hClSigmaXvsLayer[layer]->DrawNormalized();
            hClSigmaYvsLayer[layer]->DrawNormalized("same");

            hClSigmaXvsLayer[layer]->Write();
            hClSigmaYvsLayer[layer]->Write();
            cClusterSize.Write();
        }
        outFileShape.Close();
    }
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

void TrackLayerCorr(o2::its::TrackITS &ITStrack, const std::array<CompClusterExt, 7>clus, const std::array<o2::itsmft::ClusterPattern, 7>patt, TH1D* hClSizeAll, std::vector<TH1D *> hClSizeAllvsLayer, int pixThr, std::string cut)
{
    int maxClpos = std::distance(patt.begin(), std::max_element(patt.begin(), patt.end(), npix_compare));
    int npixTrigger = patt[maxClpos].getNPixels();
    
    //LOG(info) << "maxClpos: " << maxClpos;

    if ((cut == "upper") && (npixTrigger > pixThr))
    { 
        for (int layer{0}; layer < 7; layer++) // loop over layer
        {
            //LOG(info) << "Layer: " << layer;
            if (layer != maxClpos)
            {
                if (ITStrack.hasHitOnLayer(layer)) // check hit on layer
                {
                    auto &pattern = patt[layer];
                    auto npix = pattern.getNPixels();
                    //LOG(info) << "npix: " << npix;
                    hClSizeAll->Fill(npix);
                    hClSizeAllvsLayer[layer]->Fill(npix);
                }
            }
        }
    }
    else if ((cut == "lower") && (npixTrigger <= pixThr)) // npix <= 50
    {
        for (int layer{0}; layer < 7; layer++) // loop over layer
        {
            //LOG(info) << "Layer: " << layer;
            if (layer != maxClpos)
            {
                if (ITStrack.hasHitOnLayer(layer)) // check hit on layer
                {
                    auto &pattern = patt[layer];
                    auto npix = pattern.getNPixels();
                    //LOG(info) << "npix: " << npix;
                    hClSizeAll->Fill(npix);
                    hClSizeAllvsLayer[layer]->Fill(npix);
                }
            }
        }
    }
}

static bool npix_compare(o2::itsmft::ClusterPattern a, o2::itsmft::ClusterPattern b)
{
    return std::abs(a.getNPixels()) < std::abs(b.getNPixels());
}
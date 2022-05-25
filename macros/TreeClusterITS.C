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
#include "DataFormatsTPC/TrackTPC.h"

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/Propagator.h"

#include "CommonDataFormat/RangeReference.h"

#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsParameters/GRPObject.h"

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
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"

#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <DataFormatsTPC/BetheBlochAleph.h>
#include <TDatabasePDG.h>

#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;

using CompClusterExt = o2::itsmft::CompClusterExt;
using GRPObject = o2::parameters::GRPObject;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

float BetheBlochParam(const float &momentum, const float &mass)
{
    std::vector<float> parameters{0.0320980996, 19.9768009, 2.52666011e-16, 2.72123003, 6.08092022};

    // LOG(info) << momentum/mass;
    return 53 * o2::tpc::BetheBlochAleph(momentum / mass, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]) * std::pow(1, 2.3);
}

float nSigmaDeu(const float &momentum, const float &TPCSignal)
{
    float dedx = BetheBlochParam(momentum, 1.87561);
    return std::abs(TPCSignal - dedx) / (0.07 * dedx);
}

float nSigmaP(const float &momentum, const float &TPCSignal)
{
    float dedx = BetheBlochParam(momentum, TDatabasePDG::Instance()->GetParticle(2212)->Mass());
    return std::abs(TPCSignal - dedx) / (0.07 * dedx);
}

float nSigmaPi(const float &momentum, const float &TPCSignal)
{
    float dedx = BetheBlochParam(momentum, TDatabasePDG::Instance()->GetParticle(211)->Mass());

    return std::abs(TPCSignal - dedx) / (0.07 * dedx);
}

float nSigmaK(const float &momentum, const float &TPCSignal)
{
    float dedx = BetheBlochParam(momentum, TDatabasePDG::Instance()->GetParticle(321)->Mass());
    return std::abs(TPCSignal - dedx) / (0.07 * dedx);
}

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman);
void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern> &pattVec, std::vector<CompClusterExt> *ITSclus, std::vector<unsigned char> *ITSpatt, o2::itsmft::TopologyDictionary &mdict, o2::its::GeometryTGeo *gman);

void TreeClusterITS()
{
    bool isMC = false;
    int runNumber = 505658;                            // 301004 for MC
    isMC ? runNumber = 301004 : runNumber = runNumber; // 301004 for MC

    TFile outFile = TFile(Form("TreeITSClusters%i.root", runNumber), "recreate");

    TTree *MLtree = new TTree("ITStreeML", "ITStreeML");
    std::array<float, 7> clSizeArr, snPhiArr, tanLamArr, pattIDarr;
    float p, pTPC, pt, ptTPC, tgL, meanClsize, dedx, nsigmaDeu, nsigmaP, nsigmaK, nsigmaPi, tpcITSchi2;
    bool isPositive;

    MLtree->Branch("p", &p);
    MLtree->Branch("pt", &pt);
    MLtree->Branch("pTPC", &pTPC);
    MLtree->Branch("ptTPC", &ptTPC);
    MLtree->Branch("tgL", &tgL);
    MLtree->Branch("meanClsize", &meanClsize);
    MLtree->Branch("dedx", &dedx);
    MLtree->Branch("nSigmaDeu", &nsigmaDeu);
    MLtree->Branch("nSigmaP", &nsigmaP);
    MLtree->Branch("nSigmaK", &nsigmaK);
    MLtree->Branch("nSigmaPi", &nsigmaPi);
    MLtree->Branch("tpcITSchi2", &tpcITSchi2);
    MLtree->Branch("isPositive", &isPositive);

    for (int i{0}; i < 7; i++)
    {
        MLtree->Branch(Form("ClSizeL%i", i), &clSizeArr[i]);
        MLtree->Branch(Form("SnPhiL%i", i), &snPhiArr[i]);
        MLtree->Branch(Form("TanLamL%i", i), &tanLamArr[i]);
        MLtree->Branch(Form("PattIDL%i", i), &pattIDarr[i]);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    // load propagator
    TFile *f = TFile::Open("utils_clus/ccdb_grp_low_field_pos");
    auto grp = isMC ? GRPObject::loadFrom("utils_clus/o2sim_grp_MC.root") : reinterpret_cast<o2::parameters::GRPObject *>(f->Get("ccdb_object"));
    o2::base::Propagator::initFieldFromGRP(grp);
    auto *lut = o2::base::MatLayerCylSet::loadFromFile("utils_clus/matbud.root");
    o2::base::Propagator::Instance()->setMatLUT(lut);

    if (lut)
        LOG(info) << "Loaded material LUT";

    std::string resDir = isMC ? "MC" : "PBdata";

    std::string path = Form("/data/fmazzasc/its_data/%s/BPOS/%i", resDir.data(), runNumber);
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
        // if (counter > 100)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;

        std::string o2match_itstpc_file = path + "/" + dir + "/" + "o2match_itstpc.root";
        std::string o2trac_tpc_file = path + "/" + dir + "/" + "tpctracks.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "o2trac_its.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "o2clus_its.root";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fTPC = TFile::Open(o2trac_tpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

        if (!fITS || !fITSTPC || !fITSclus || !fTPC)
            continue;

        auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
        auto treeTPC = (TTree *)fTPC->Get("tpcrec");
        auto treeITS = (TTree *)fITS->Get("o2sim");
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        // Tracks
        std::vector<o2::dataformats::TrackTPCITS> *TPCITStracks = nullptr;
        std::vector<o2::its::TrackITS> *ITStracks = nullptr;
        std::vector<o2::tpc::TrackTPC> *TPCtracks = nullptr;

        std::vector<int> *ITSTrackClusIdx = nullptr;

        // Clusters
        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        // Setting branches
        treeITS->SetBranchAddress("ITSTrack", &ITStracks);
        treeTPC->SetBranchAddress("TPCTracks", &TPCtracks);
        treeITSTPC->SetBranchAddress("TPCITS", &TPCITStracks);
        treeITS->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);
        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {

            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame) || !treeTPC->GetEvent(frame))
                continue;

            std::vector<ITSCluster> ITSClusXYZ;
            ITSClusXYZ.reserve((*ITSclus).size());
            gsl::span<const unsigned char> spanPatt{*ITSpatt};
            auto pattIt = spanPatt.begin();
            o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, ITSClusXYZ, &mdict);

            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {

                auto &ITSTPCtrack = TPCITStracks->at(iTrack);
                auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());
                auto &TPCtrack = TPCtracks->at(ITSTPCtrack.getRefTPC());

                std::array<CompClusterExt, 7> TrackClus;
                std::array<ITSCluster, 7> TrackClusXYZ;
                std::array<unsigned int, 7> TrackPattID;
                std::array<o2::itsmft::ClusterPattern, 7> TrackPatt;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &patt = pattVec[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &clusXYZ = ITSClusXYZ[(*ITSTrackClusIdx)[firstClus + icl]];

                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus[layer] = clus;
                    TrackClusXYZ[layer] = clusXYZ;
                    TrackPattID[layer] = clus.getPatternID();
                    TrackPatt[layer] = patt;
                }

                for (int layer{0}; layer < 7; layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {

                        clSizeArr[layer] = TrackPatt[layer].getNPixels();
                    }
                    else
                        clSizeArr[layer] = -10;
                }

                if (ITSTPCtrack.getChi2Match() < 10)
                {

                    float mean = 0, norm = 0;
                    for (unsigned int i{0}; i < clSizeArr.size(); i++)
                    {
                        if (clSizeArr[i] > 0)
                        {
                            mean += clSizeArr[i];
                            norm += 1;
                        }
                    }
                    mean /= norm;
                    mean *= std::sqrt(1. / (1 + ITSTPCtrack.getTgl() * ITSTPCtrack.getTgl()));

                    p = ITSTPCtrack.getP();
                    pt = ITSTPCtrack.getPt();
                    p = TPCtrack.getP();
                    pt = TPCtrack.getPt();
                    isPositive = ITSTPCtrack.getSign() == 1;
                    tgL = ITSTPCtrack.getTgl();
                    meanClsize = mean;
                    dedx = TPCtrack.getdEdx().dEdxTotTPC;

                    nsigmaDeu = nSigmaDeu(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                    nsigmaP = nSigmaP(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                    nsigmaPi = nSigmaPi(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                    nsigmaK = nSigmaK(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);

                    tpcITSchi2 = ITSTPCtrack.getChi2Match();

                    for (unsigned int layer{0}; layer < clSizeArr.size(); layer++)
                    {
                        if (ITStrack.hasHitOnLayer(layer))
                        {
                            auto &clusXYZ = TrackClusXYZ[layer];
                            pattIDarr[layer] = TrackPattID[layer];
                            if (propagateToClus(clusXYZ, ITSTPCtrack, gman))
                            {
                                snPhiArr[layer] = ITSTPCtrack.getSnp();
                                tanLamArr[layer] = ITSTPCtrack.getTgl();
                            }

                            else
                            {
                                snPhiArr[layer] = -10;
                                tanLamArr[layer] = -10;
                            }
                        }
                        else
                        {
                            pattIDarr[layer] = -10;
                            snPhiArr[layer] = -10;
                            tanLamArr[layer] = -10;
                        }
                    }
                    MLtree->Fill();
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
    outFile.cd();

    MLtree->Write();
    outFile.Close();
}

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman)
{

    auto corrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;
    auto propInstance = o2::base::Propagator::Instance();
    float alpha = gman->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    int layer{gman->getLayer(clus.getSensorID())};

    if (!track.rotate(alpha))
        return false;

    if (!propInstance->propagateToX(track, x, propInstance->getNominalBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, corrType))
        return false;

    return true;
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
            // LOG(info) << "is invalid pattern: "<< (pattID == o2::itsmft::CompCluster::InvalidPatternID);
            // LOG(info) << "is group: "<< mdict.isGroup(pattID);
            patt.acquirePattern(pattIt);
            npix = patt.getNPixels();
        }
        else
        {

            npix = mdict.getNpixels(pattID);
            patt = mdict.getPattern(pattID);
        }
        pattVec.push_back(patt);
    }
}
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
    return (TPCSignal - dedx) / (0.07 * dedx);
}

float nSigma(const float &momentum, const float &TPCSignal, int pdgCode)
{
    float dedx = BetheBlochParam(momentum, TDatabasePDG::Instance()->GetParticle(pdgCode)->Mass());
    return (TPCSignal - dedx) / (0.08 * dedx);
}

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman);
void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern> &pattVec, std::vector<CompClusterExt> *ITSclus, std::vector<unsigned char> *ITSpatt, o2::itsmft::TopologyDictionary &mdict, o2::its::GeometryTGeo *gman);

void ITSTPCClusterTreeBuilder()
{
    bool isMC = false;
    int runNumber = 520143;        // MAY: 517618,  JUNE: 518543, OCT: 505658, JUL:
    std::string runPeriod = "JUL"; // could be either OCT, MAY, JUN, JUL

    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    TFile *fGRP = TFile::Open("../utils/ccdb_grp_low_field_pos");
    auto grp = reinterpret_cast<o2::parameters::GRPObject *>(fGRP->Get("ccdb_object"));
    fGRP->Close();

    std::string pathDir;

    if (runPeriod == "OCT")
    {
        mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "../utils/ITSdictionary.bin"));
        pathDir = "/data/shared/ITS/OCT/CTFS";
    }
    else if (runPeriod == "MAY")
    {
        auto fdic = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
        mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(fdic.Get("ccdb_object")));
        fdic.Close();
        pathDir = "/data/shared/ITS/MAY/CTFS";
    }
    else if (runPeriod == "JUN")
    {
        auto fdic = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
        mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(fdic.Get("ccdb_object")));
        fdic.Close();
        pathDir = "/data/shared/ITS/JUN/CTFS";
    }

    else if (runPeriod == "JUL")
    {
        auto fdic = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
        mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(fdic.Get("ccdb_object")));
        fdic.Close();
        pathDir = "/data/shared/ITS/JUL/pp13TeV/apass/LHC22f";
    }

    else
    {
        LOG(fatal) << "Run period not recognized";
    }

    TFile outFile = TFile(Form("../results/ITSTPCClusterTree%i.root", runNumber), "recreate");
    TTree *MLtree = new TTree("ITStreeML", "ITStreeML");
    std::array<float, 7> clSizeArr, snPhiArr, tanLamArr, pattIDarr;
    float p, pTPC, pt, ptTPC, pITS, ptITS, tgL, clSizeCosLam, dedx, nsigmaDeu, nsigmaP, nsigmaK, nsigmaPi, nsigmaE,tpcITSchi2, itsChi2, tpcChi2;
    bool isPositive;
    int nClusITS;
    int nClusTPC;
    int rofBC;


    MLtree->Branch("p", &p);
    MLtree->Branch("pt", &pt);
    MLtree->Branch("pTPC", &pTPC);
    MLtree->Branch("ptTPC", &ptTPC);
    MLtree->Branch("pITS", &pITS);
    MLtree->Branch("ptITS", &ptITS);
    MLtree->Branch("rofBC", &rofBC);
    MLtree->Branch("tgL", &tgL);
    MLtree->Branch("clSizeCosLam", &clSizeCosLam);
    MLtree->Branch("dedx", &dedx);
    MLtree->Branch("nSigmaDeu", &nsigmaDeu);
    MLtree->Branch("nSigmaP", &nsigmaP);
    MLtree->Branch("nSigmaK", &nsigmaK);
    MLtree->Branch("nSigmaPi", &nsigmaPi);
    MLtree->Branch("nSigmaE", &nsigmaE);
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
    o2::base::GeometryManager::loadGeometry("../utils/o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

    // load propagator
    o2::base::Propagator::initFieldFromGRP(grp);

    auto *lut = o2::base::MatLayerCylSet::loadFromFile("../utils/matbud.root");
    o2::base::Propagator::Instance()->setMatLUT(lut);

    if (lut)
        LOG(info) << "Loaded material LUT";

    std::string path = Form("%s/%i", pathDir.data(), runNumber);
    LOG(info) << "Reading from " << path;
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
        // if (counter > 50)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;

        std::string o2match_itstpc_file = path + "/" + dir + "/" + "root_archive.zip#o2match_itstpc.root";
        std::string o2trac_tpc_file = path + "/" + dir + "/" + "root_archive.zip#tpctracks.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "root_archive.zip#o2trac_its.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "root_archive.zip#o2clus_its.root";

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
        std::vector<o2::itsmft::ROFRecord> *ROFits = nullptr;
        std::vector<o2::tpc::TrackTPC> *TPCtracks = nullptr;

        std::vector<int> *ITSTrackClusIdx = nullptr;

        // Clusters
        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        // Setting branches
        treeITS->SetBranchAddress("ITSTrack", &ITStracks);
        treeITS->SetBranchAddress("ITSTracksROF", &ROFits);
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
            ITSClusXYZ.reserve(ITSclus->size());
            gsl::span<const unsigned char> spanPatt{*ITSpatt};
            auto pattIt = spanPatt.begin();
            o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, ITSClusXYZ, &mdict);

            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {

                auto &ITSTPCtrack = TPCITStracks->at(iTrack);
                // LOG(info) << "Source: " << ITSTPCtrack.getRefITS().asString();
                if (ITSTPCtrack.getRefITS().getSource() == 24) //excluding Afterburned tracks
                {
                    // LOG(info) << "ITS track: " << iTrack;
                    continue;
                }
                auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());

                for (auto& rof : *ROFits) {
                    if (ITSTPCtrack.getRefITS().getIndex() <= rof.getFirstEntry()) {
                        rofBC = rof.getBCData().bc;
                        // LOG(info) << "rofBC: " << rofBC;
                        break;
                    }
                }

                auto &TPCtrack = TPCtracks->at(ITSTPCtrack.getRefTPC());

                std::array<CompClusterExt, 7> TrackClus;
                std::array<ITSCluster, 7> TrackClusXYZ;
                std::array<unsigned int, 7> TrackPattID;
                std::array<o2::itsmft::ClusterPattern, 7> TrackPatt;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();
                nClusITS = ncl;

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

                if (ITSTPCtrack.getChi2Match() < 10 && ITSTPCtrack.getLTIntegralOut().getL() > 250)
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
                    pTPC = TPCtrack.getP();
                    ptTPC = TPCtrack.getPt();
                    pITS = ITStrack.getP();
                    ptITS = ITStrack.getPt();
                    isPositive = ITSTPCtrack.getSign() == 1;
                    tgL = ITSTPCtrack.getTgl();
                    clSizeCosLam = mean;
                    dedx = TPCtrack.getdEdx().dEdxTotTPC;
                    nClusTPC = TPCtrack.getNClusters();

                    itsChi2 = ITStrack.getChi2();
                    tpcChi2 = TPCtrack.getChi2();

                    nsigmaDeu = nSigmaDeu(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                    nsigmaP = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 2212);
                    nsigmaK = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 321);
                    nsigmaPi = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 211);
                    nsigmaE = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 11);


                    tpcITSchi2 = ITSTPCtrack.getChi2Match();

                    for (unsigned int layer{0}; layer < clSizeArr.size(); layer++)
                    {
                        if (ITStrack.hasHitOnLayer(layer))
                        {
                            auto &clusXYZ = TrackClusXYZ[layer];
                            pattIDarr[layer] = TrackPattID[layer];

                            // LOG(info) << "Layer: " << layer << " pattID: " << pattIDarr[layer] << " clSize: " << clSizeArr[layer];
                            // if (propagateToClus(clusXYZ, ITSTPCtrack, gman))
                            // {
                            //     snPhiArr[layer] = ITSTPCtrack.getSnp();
                            //     tanLamArr[layer] = ITSTPCtrack.getTgl();
                            // }

                            // else
                            // {
                            snPhiArr[layer] = -10;
                            tanLamArr[layer] = -10;
                            // }
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
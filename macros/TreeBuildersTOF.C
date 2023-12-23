#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"

#include "CommonDataFormat/RangeReference.h"
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
#include "ReconstructionDataFormats/MatchInfoTOF.h"
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

#include "../utils/ClusterStudyUtils.h"

#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using PID = o2::track::PID;

using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;

using CompClusterExt = o2::itsmft::CompClusterExt;
using GRPObject = o2::parameters::GRPObject;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

void TreeBuilderTOF(int runNumber = 520147, std::string runPeriod = "JUL", 
std::string OutDir = "../results/")
{
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    o2::parameters::GRPObject *grp;
    std::string pathDir;

    GetTopologyDictionary(mdict, grp, runPeriod, pathDir);

    struct particle
    {
        int pdg;
        float p;
        float pt;
        float eta;
        float phi;
        float dedx;
        float nsigmaDeu;
        float nsigmaP;
        float nsigmaK;
        float nsigmaPi;
        float nsigmaE;
        float tpcITSchi2;
        float itsChi2;
        float tpcChi2;
        float tgL;
        float snPhi;
        float clSize;
        float clSizeCosLam;
        float pattID;
        float rofBC;
        float nClusITS;
        float nClusTPC;
        float pITS;
        float ptITS;
        float pTPC;
        float ptTPC;
        float isPositive;
        std::array<float, 7> clSizes;
        std::array<float, 7> snPhis;
        std::array<float, 7> tanLams;
        std::array<float, 7> pattIDs;
        float TOFdeltaT = -1;
        float TOFsignal = -1;
        float TOFtrLength = -1;
        float TOFChi2 = -1;
        bool isAB = false;
    };

    std::vector<particle> particles;
    // finally, loop over the particles and fill the tree
    TFile outFile = TFile(Form("%sTreeBuilderTOF_%i.root", OutDir.data(), runNumber), "RECREATE");
    TTree *MLtree = new TTree("ITStreeML", "ITStreeML");
    MLtree->Branch("particles", &particles);

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
        std::string o2match_itstpctof_file = path + "/" + dir + "/" + "root_archive.zip#o2match_tof_itstpc.root";
        std::string o2trac_tpc_file = path + "/" + dir + "/" + "root_archive.zip#tpctracks.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "root_archive.zip#o2trac_its.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "root_archive.zip#o2clus_its.root";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fITSTPCTOF = TFile::Open(o2match_itstpctof_file.data());
        auto fTPC = TFile::Open(o2trac_tpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

        if (!fITS || !fITSTPC || !fITSclus || !fTPC || !fITSTPCTOF)
            continue;

        auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
        auto treeITSTPCTOF = (TTree *)fITSTPCTOF->Get("matchTOF");
        auto treeTPC = (TTree *)fTPC->Get("tpcrec");
        auto treeITS = (TTree *)fITS->Get("o2sim");
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        // Tracks
        std::vector<o2::dataformats::TrackTPCITS> *TPCITStracks = nullptr;
        std::vector<o2::dataformats::MatchInfoTOF> *TOFmatch = nullptr;
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
        treeITSTPCTOF->SetBranchAddress("TOFMatchInfo", &TOFmatch);

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {

            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame) || !treeTPC->GetEvent(frame) || !treeITSTPCTOF->GetEvent(frame))
                continue;

            particles.clear();

            std::vector<ITSCluster> ITSClusXYZ;
            ITSClusXYZ.reserve(ITSclus->size());
            gsl::span<const unsigned char> spanPatt{*ITSpatt};
            auto pattIt = spanPatt.begin();
            o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, ITSClusXYZ, &mdict);

            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {
                particle part;

                auto &ITSTPCtrack = TPCITStracks->at(iTrack);
                if (ITSTPCtrack.getRefITS().getSource() == 24) // excluding Afterburned tracks
                {
                    part.isAB = true;
                    particles.push_back(part);
                    continue;
                }
                auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());

                for (auto &rof : *ROFits)
                {
                    if (ITSTPCtrack.getRefITS().getIndex() <= rof.getFirstEntry())
                    {
                        part.rofBC = rof.getBCData().bc;
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
                part.nClusITS = ncl;

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

                        part.clSizes[layer] = TrackPatt[layer].getNPixels();
                    }
                    else
                        part.clSizes[layer] = -10;
                }

                float mean = 0, norm = 0;
                for (unsigned int i{0}; i < part.clSizes.size(); i++)
                {
                    if (part.clSizes[i] > 0)
                    {
                        mean += part.clSizes[i];
                        norm += 1;
                    }
                }
                mean /= norm;
                mean *= std::sqrt(1. / (1 + ITSTPCtrack.getTgl() * ITSTPCtrack.getTgl()));

                part.p = ITSTPCtrack.getP();
                part.pt = ITSTPCtrack.getPt();
                part.pTPC = TPCtrack.getP();
                part.ptTPC = TPCtrack.getPt();
                part.pITS = ITStrack.getP();
                part.ptITS = ITStrack.getPt();
                part.isPositive = ITSTPCtrack.getSign() == 1;
                part.tgL = ITSTPCtrack.getTgl();
                part.clSizeCosLam = mean;
                part.dedx = TPCtrack.getdEdx().dEdxTotTPC;
                part.nClusTPC = TPCtrack.getNClusters();

                part.itsChi2 = ITStrack.getChi2();
                part.tpcChi2 = TPCtrack.getChi2();

                part.nsigmaDeu = nSigmaDeu(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                part.nsigmaP = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 2212);
                part.nsigmaK = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 321);
                part.nsigmaPi = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 211);
                part.nsigmaE = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 11);

                part.tpcITSchi2 = ITSTPCtrack.getChi2Match();

                for (unsigned int layer{0}; layer < part.clSizes.size(); layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {
                        auto &clusXYZ = TrackClusXYZ[layer];
                        part.pattIDs[layer] = TrackPattID[layer];
                        if (propagateToClus(clusXYZ, ITSTPCtrack, gman))
                        {
                            part.snPhis[layer] = ITSTPCtrack.getSnp();
                            part.tanLams[layer] = ITSTPCtrack.getTgl();
                        }

                        else
                        {
                            part.snPhis[layer] = -10;
                            part.tanLams[layer] = -10;
                        }
                    }
                    else
                    {
                        part.pattIDs[layer] = -10;
                        part.snPhis[layer] = -10;
                        part.tanLams[layer] = -10;
                    }
                }
                particles.push_back(part);
            }
            // now loop over the TOF information
            for (unsigned int iTof = 0; iTof < TOFmatch->size(); iTof++)
            {
                auto &tofInfo = TOFmatch->at(iTof);
                auto &part = particles[tofInfo.getTrackIndex()];
                part.TOFdeltaT = tofInfo.getDeltaT();
                part.TOFsignal = tofInfo.getSignal();
                part.TOFtrLength = tofInfo.getLTIntegralOut().getL();
                part.TOFChi2 = tofInfo.getChi2();
            }
            for (auto &part : particles)
            {
                if (part.isAB == true)
                    continue;

                MLtree->Fill();
            }
        }
    }

    MLtree->Write();
    outFile.Close();
}
#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/Cascade.h"
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
double calcV0alpha(const V0 &v0);
double calcV0qt(const V0 &v0);
double calcMass(const V0 &v0, PID v0PID);

void V0ClusterTreeBuilder()
{

    int runNumber = 520147;  //520147, 520178, 520150

    TFile outFile = TFile(Form("../results/V0TreePIDITS_%i.root", runNumber), "recreate");

    TTree *DauTree = new TTree("DauTree", "DauTree");
    TTree *V0Tree = new TTree("V0Tree", "V0Tree");
    std::array<float, 7> clSizeArr, snPhiArr, tanLamArr, pattIDarr;
    float p, pTPC, pt, ptTPC, pITS, ptITS, rofBC, tgL, clSizeCosLam, dedx, tpcITSchi2;
    float nsigmaDeu, nsigmaP, nsigmaK, nsigmaPi, nsigmaE;
    int V0ind = 0;

    // V0 tree elements
    float V0radius, V0CosPA, V0ArmenterosAlpha, V0ArmenterosQt, V0p;
    float photMassHyp, k0sMassHyp, lamMassHyp;
    float nSigmaPosDauP = -10, nSigmaNegDauP = -10, nSigmaPosDauPi = -10, nSigmaNegDauPi = -10, nSigmaPosDauE = -10, nSigmaNegDauE = -10;
    bool isPositive;

    V0Tree->Branch("V0radius", &V0radius);
    V0Tree->Branch("V0CosPA", &V0CosPA);
    V0Tree->Branch("V0ArmenterosAlpha", &V0ArmenterosAlpha);
    V0Tree->Branch("V0ArmenterosQt", &V0ArmenterosQt);
    V0Tree->Branch("photMassHyp", &photMassHyp);
    V0Tree->Branch("k0sMassHyp", &k0sMassHyp);
    V0Tree->Branch("lamMassHyp", &lamMassHyp);
    V0Tree->Branch("v0Ind", &V0ind);
    V0Tree->Branch("V0p", &V0p);
    V0Tree->Branch("nSigmaPosDauP", &nSigmaPosDauP);
    V0Tree->Branch("nSigmaNegDauP", &nSigmaNegDauP);
    V0Tree->Branch("nSigmaPosDauPi", &nSigmaPosDauPi);
    V0Tree->Branch("nSigmaNegDauPi", &nSigmaNegDauPi);
    V0Tree->Branch("nSigmaPosDauE", &nSigmaPosDauE);
    V0Tree->Branch("nSigmaNegDauE", &nSigmaNegDauE);

    DauTree->Branch("p", &p);
    DauTree->Branch("pt", &pt);
    DauTree->Branch("pTPC", &pTPC);
    DauTree->Branch("ptTPC", &ptTPC);
    DauTree->Branch("pITS", &pITS);
    DauTree->Branch("ptITS", &ptITS);
    DauTree->Branch("rofBC", &rofBC);
    DauTree->Branch("tgL", &tgL);
    DauTree->Branch("clSizeCosLam", &clSizeCosLam);
    DauTree->Branch("dedx", &dedx);
    DauTree->Branch("nSigmaDeu", &nsigmaDeu);
    DauTree->Branch("nSigmaP", &nsigmaP);
    DauTree->Branch("nSigmaK", &nsigmaK);
    DauTree->Branch("nSigmaPi", &nsigmaPi);
    DauTree->Branch("nSigmaE", &nsigmaE);
    DauTree->Branch("tpcITSchi2", &tpcITSchi2);
    DauTree->Branch("isPositive", &isPositive);
    DauTree->Branch("v0Ind", &V0ind);

    for (int i{0}; i < 7; i++)
    {
        DauTree->Branch(Form("ClSizeL%i", i), &clSizeArr[i]);
        DauTree->Branch(Form("SnPhiL%i", i), &snPhiArr[i]);
        DauTree->Branch(Form("TanLamL%i", i), &tanLamArr[i]);
        DauTree->Branch(Form("PattIDL%i", i), &pattIDarr[i]);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("/data/fmazzasc/its_data/sim/lambda/tf1/o2sim_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    o2::itsmft::ChipMappingITS chipMapping;
    auto fdic = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
    mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(fdic.Get("ccdb_object")));

    // load propagator
    TFile *f = TFile::Open("/data/fmazzasc/its_data/sim/lambda/tf1/o2sim_grp.root");
    auto grp = reinterpret_cast<o2::parameters::GRPObject *>(f->Get("ccdb_object"));
    o2::base::Propagator::initFieldFromGRP(grp);
    auto *lut = o2::base::MatLayerCylSet::loadFromFile("../utils/matbud.root");
    o2::base::Propagator::Instance()->setMatLUT(lut);

    if (lut)
        LOG(info) << "Loaded material LUT";

    std::string path = Form("/data/shared/ITS/JUL/pp13TeV/apass/LHC22f/%i/", runNumber);
    TSystemDirectory dir("MyDir", path.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs;
    for (auto fileObj : *files)
    {
        std::string file = ((TSystemFile *)fileObj)->GetName();
        if (file.substr(0, 6) == "o2_ctf")
        {
            dirs.push_back(file);
        }
    }

    std::sort(dirs.begin(), dirs.end());

    int counter = 0;

    for (auto &dir : dirs)
    {
        // if (counter > 100)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;

        std::string secondary_vertex_file = path + "/" + dir + "/" + "root_archive.zip#o2_secondary_vertex.root";
        std::string o2match_itstpc_file = path + "/" + dir + "/" + "root_archive.zip#o2match_itstpc.root";
        std::string o2match_tof_itstpc_file = path + "/" + dir + "/" + "root_archive.zip#o2match_tof_itstpc.root";

        std::string o2trac_tpc_file = path + "/" + dir + "/" + "root_archive.zip#tpctracks.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "root_archive.zip#o2trac_its.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "root_archive.zip#o2clus_its.root";

        // Files
        auto fV0 = TFile::Open(secondary_vertex_file.data());
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fITSTPCTOF = TFile::Open(o2match_tof_itstpc_file.data());

        auto fTPC = TFile::Open(o2trac_tpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());

        if (!fITS || !fITSTPC || !fITSclus || !fTPC || !fV0 || !fITSTPCTOF)
            continue;

        auto treeV0 = (TTree *)(fV0->Get("o2sim"));
        auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
        auto treeITSTPCTOF = (TTree *)fITSTPCTOF->Get("matchTOF");
        auto treeTPC = (TTree *)fTPC->Get("tpcrec");
        auto treeITS = (TTree *)fITS->Get("o2sim");
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        // Tracks
        std::vector<V0> *V0tracks = nullptr;
        std::vector<o2::dataformats::TrackTPCITS> *TPCITStracks = nullptr;
        std::vector<o2::its::TrackITS> *ITStracks = nullptr;
        std::vector<o2::tpc::TrackTPC> *TPCtracks = nullptr;
        std::vector<o2::dataformats::MatchInfoTOF> *TOFtracks = nullptr;

        std::vector<int> *ITSTrackClusIdx = nullptr;
        std::vector<o2::itsmft::ROFRecord> *ROFits = nullptr;


        // Clusters
        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        // Setting branches
        treeV0->SetBranchAddress("V0s", &V0tracks);

        treeTPC->SetBranchAddress("TPCTracks", &TPCtracks);
        treeITSTPC->SetBranchAddress("TPCITS", &TPCITStracks);

        treeITS->SetBranchAddress("ITSTrack", &ITStracks);
        treeITS->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);
        treeITS->SetBranchAddress("ITSTracksROF", &ROFits);

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
        treeITSTPCTOF->SetBranchAddress("TOFMatchInfo", &TOFtracks);

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {
            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame) ||
                !treeTPC->GetEvent(frame) || !treeV0->GetEvent(frame) || !treeITSTPCTOF->GetEvent(frame))
                continue;
            // LOG(info) << "Processing frame: " << frame;

            std::vector<ITSCluster> ITSClusXYZ;
            ITSClusXYZ.reserve((*ITSclus).size());
            gsl::span<const unsigned char> spanPatt{*ITSpatt};
            auto pattIt = spanPatt.begin();

            o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, ITSClusXYZ, &mdict);

            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);

            for (unsigned int iV0{0}; iV0 < V0tracks->size(); ++iV0)
            {

                auto &v0 = V0tracks->at(iV0);
                V0ind++;
                V0CosPA = v0.getCosPA();
                V0radius = v0.calcR2();
                V0ArmenterosAlpha = calcV0alpha(v0);
                V0ArmenterosQt = calcV0qt(v0);
                V0p = v0.getP();
                photMassHyp = v0.calcMass2(PID::Electron, PID::Electron);
                k0sMassHyp = calcMass(v0, PID::K0);
                lamMassHyp = calcMass(v0, PID::Lambda);

                for (int v0Dau{0}; v0Dau < 2; v0Dau++)
                {
                    auto v0DauID = v0.getProngID(v0Dau);
                    auto trackIndex = v0DauID.getIndex();
                    if (v0DauID.getSourceName() != "ITS-TPC" && v0DauID.getSourceName() != "ITS-TPC-TOF")
                        continue;

                    if (v0DauID.getSourceName() == "ITS-TPC-TOF")
                    {
                        auto &tofInfo = TOFtracks->at(trackIndex);
                        auto trackRef = tofInfo.getTrackRef();
                        if (trackRef.getSourceName() != "ITS-TPC")
                            continue;
                        trackIndex = tofInfo.getTrackIndex();
                    }

                    auto &ITSTPCtrack = TPCITStracks->at(trackIndex);
                    auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());
                    auto &TPCtrack = TPCtracks->at(ITSTPCtrack.getRefTPC());

                    for (auto &rof : *ROFits)
                    {
                        if (int(ITSTPCtrack.getRefITS().getIndex()) <= rof.getFirstEntry())
                        {
                            rofBC = rof.getBCData().bc;
                            break;
                        }
                    }

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

                    if (ITSTPCtrack.getChi2Match() > 10)
                        continue;

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
                    tpcITSchi2 = ITSTPCtrack.getChi2Match();

                    nsigmaDeu = nSigmaDeu(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                    nsigmaP = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 2212);
                    nsigmaK = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 321);
                    nsigmaPi = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 211);
                    nsigmaE = nSigma(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC, 11);

                    if (isPositive)
                    {
                        nSigmaPosDauP = nsigmaP;
                        nSigmaPosDauPi = nsigmaPi;
                        nSigmaPosDauE = nsigmaE;
                    }
                    else
                    {
                        nSigmaNegDauP = nsigmaP;
                        nSigmaNegDauPi = nsigmaPi;
                        nSigmaNegDauE = nsigmaE;
                    }

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
                    DauTree->Fill();
                }
                V0Tree->Fill();
            }
        }
        treeV0->ResetBranchAddresses();
        treeITS->ResetBranchAddresses();
        treeITSTPC->ResetBranchAddresses();
        treeITSclus->ResetBranchAddresses();
        fITS->Close();
        fITSTPC->Close();
        fITSclus->Close();
    }
    outFile.cd();
    DauTree->Write();
    V0Tree->Write();
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
        auto &clus = ITSclus->at(iClus);
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
        pattVec.push_back(patt);
    }
}

double calcV0alpha(const V0 &v0)
{
    std::array<float, 3> fV0mom, fPmom, fNmom = {0, 0, 0};
    v0.getProng(0).getPxPyPzGlo(fPmom);
    v0.getProng(1).getPxPyPzGlo(fNmom);
    v0.getPxPyPzGlo(fV0mom);

    TVector3 momNeg(fNmom[0], fNmom[1], fNmom[2]);
    TVector3 momPos(fPmom[0], fPmom[1], fPmom[2]);
    TVector3 momTot(fV0mom[0], fV0mom[1], fV0mom[2]);

    Double_t lQlNeg = momNeg.Dot(momTot) / momTot.Mag();
    Double_t lQlPos = momPos.Dot(momTot) / momTot.Mag();

    return (lQlPos - lQlNeg) / (lQlPos + lQlNeg);
}

double calcV0qt(const V0 &v0)
{
    std::array<float, 3> fV0mom, fPmom, fNmom = {0, 0, 0};
    v0.getProng(0).getPxPyPzGlo(fPmom);
    v0.getProng(1).getPxPyPzGlo(fNmom);
    v0.getPxPyPzGlo(fV0mom);

    TVector3 momNeg(fNmom[0], fNmom[1], fNmom[2]);
    TVector3 momTot(fV0mom[0], fV0mom[1], fV0mom[2]);
    Double_t dp = momNeg.Dot(momTot);

    return std::sqrt(momNeg.Mag2() - dp * dp / momTot.Mag2());
}

double calcMass(const V0 &v0, PID v0PID)
{
    std::array<double, 2> dauMass{TDatabasePDG::Instance()->GetParticle(211)->Mass(), TDatabasePDG::Instance()->GetParticle(211)->Mass()};
    if (v0PID == PID::Lambda)
        dauMass = {TDatabasePDG::Instance()->GetParticle(2212)->Mass(), TDatabasePDG::Instance()->GetParticle(211)->Mass()};

    if (calcV0alpha(v0) < 0)
        std::swap(dauMass[0], dauMass[1]);

    std::vector<o2::dataformats::V0::Track> dauTracks = {v0.getProng(0), v0.getProng(1)};
    TLorentzVector moth, prong;
    std::array<float, 3> p;
    for (int i = 0; i < 2; i++)
    {
        auto &track = dauTracks[i];
        auto &mass = dauMass[i];
        track.getPxPyPzGlo(p);
        prong.SetVectM({p[0], p[1], p[2]}, mass);
        moth += prong;
    }
    return moth.M();
}
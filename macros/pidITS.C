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

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman, float Bz = -5.);

void pidITS()
{

    TH2D *hSplines = new TH2D("ITS splines ", ";#it{p}^{ITS-TPC} (GeV/#it{c}); #LT Cluster size #GT #times Cos(#lambda) ; Counts", 300, 0, 2, 60, 0.5, 12.5);
    TH2D *hSplinesSA = new TH2D("ITS splines ITS SA ", ";#it{p}^ITS (GeV/#it{c}); #LT Cluster size #GT #times Cos(#lambda) ; Counts", 300, 0, 2, 60, 0.5, 12.5);
    TH1D *hClSizeDeu = new TH1D("Average Cl size for deuterons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);
    TH1D *hClSizeP = new TH1D("Average Cl size for protons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);
    TH1D *hClSizeK = new TH1D("Average Cl size for kaons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);

    TH1D *hClSizePi = new TH1D("Average Cl size for pi", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);
    TH1D *hTotClSizeP = new TH1D("Cl size for protons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 13, 0.5, 13.5);
    TH1D *hTotClSizePi = new TH1D("Cl size for pi", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 13, 0.5, 13.5);
    TH2D *hSplinesTPC = new TH2D("TPC splines ", ";#it{p}^{ITS-TPC} (GeV/#it{c}); TPC Signal ; Counts", 300, 0.05, 2, 300, 30.5, 600.5);
    TH1D *hPtRes = new TH1D("pT resolution ", ";(#it{p}^{ITS-SA} - #it{p}^{ITS-TPC})/#it{p}^{ITS-TPC}; Counts", 80, -1, 1);
    TH2D *hChi2ClSize = new TH2D("Cluster size vs Chi2", "; Matching #chi^{2}; #LT Cluster size #GT #times Cos(#lambda) ; Counts", 300, 0, 600, 40, 0.5, 12.5);

    TFile outFile = TFile("pid.root", "recreate");

    TTree *MLtree = new TTree("ITStreeML", "ITStreeML");
    std::array<float, 7> clSizeArr, snPhiArr;
    float p, pt, tgL, meanClsize, dedx, label;
    for (int i{0}; i < 7; i++)
    {
        MLtree->Branch(Form("ClSizeL%i", i), &clSizeArr[i]);
        MLtree->Branch(Form("SnPhiL%i", i), &snPhiArr[i]);
    }

    MLtree->Branch("p", &p);
    MLtree->Branch("pt", &pt);
    MLtree->Branch("tgL", &tgL);
    MLtree->Branch("meanClsize", &meanClsize);
    MLtree->Branch("dedx", &dedx);
    MLtree->Branch("label", &label);

    bool usePaki = false;

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));
    // Propagator
    const auto grp = GRPObject::loadFrom();
    // load propagator
    o2::base::Propagator::initFieldFromGRP(grp);
    auto *lut = o2::base::MatLayerCylSet::loadFromFile("matbud.root");
    o2::base::Propagator::Instance()->setMatLUT(lut);

    if (lut)
        LOG(info) << "Loaded material LUT";

    std::string path = "/data/fmazzasc/its_data/merge";
    TSystemDirectory dir("MyDir", path.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs;
    for (auto fileObj : *files)
    {
        std::string file = ((TSystemFile *)fileObj)->GetName();
        if (file.substr(0, 6) == "o2_ctf")
            dirs.push_back(file);
    }

    int counter = 0;

    for (auto &dir : dirs)
    {
        // if (counter > 10)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;

        std::string o2match_itstpc_file = path + "/" + dir + "/" + "o2match_itstpc.root";
        std::string o2trac_its_file = path + "/" + dir + "/" + "o2trac_its.root";
        std::string o2trac_tpc_file = path + "/" + dir + "/" + "tpc_tracks.root";
        std::string o2clus_its_file = path + "/" + dir + "/" + "o2clus_its.root";

        // Files
        auto fITSTPC = TFile::Open(o2match_itstpc_file.data());
        auto fTPC = TFile::Open(o2trac_tpc_file.data());
        auto fITS = TFile::Open(o2trac_its_file.data());
        auto fITSclus = TFile::Open(o2clus_its_file.data());
        if (!fITS || !fTPC || !fITSTPC || !fITSclus)
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

        std::vector<float> pakiWeights{0.11, 0.7, 0.7, 0.75, 0.7, 0.6, 0.09};
        float sumPaki = std::accumulate(pakiWeights.begin(), pakiWeights.end(), 0.);
        bool isFileCorrupted = false;

        for (int frame = 0; frame < treeITSTPC->GetEntriesFast(); frame++)
        {
            if (isFileCorrupted)
                break;

            if (!treeITSTPC->GetEvent(frame) || !treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame) || !treeTPC->GetEvent(frame))
                continue;

            auto pattIt2 = ITSpatt->cbegin();

            std::vector<ITSCluster> ITSClusXYZ;
            ITSClusXYZ.reserve((*ITSclus).size());
            gsl::span<const unsigned char> spanPatt{*ITSpatt};
            auto pattIt = spanPatt.begin();
            o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, ITSClusXYZ, mdict);

            for (unsigned int iTrack{0}; iTrack < TPCITStracks->size(); ++iTrack)
            {

                auto &ITSTPCtrack = TPCITStracks->at(iTrack);

                if (int(ITSTPCtrack.getRefITS().getIndex()) > int(ITStracks->size()) - 1 || int(ITSTPCtrack.getRefTPC().getIndex()) > int(TPCtracks->size()) - 1)
                {
                    LOG(info) << "Frame:" << frame << ", Ind exception: " << ITSTPCtrack.getRefITS().getIndex() << ", ITS Track size" << ITStracks->size();
                    LOG(info) << "Frame:" << frame << ", Ind exception: " << ITSTPCtrack.getRefITS().getIndex() << ", TPC Track size" << ITStracks->size();
                    isFileCorrupted = true;
                    break;
                }

                auto &ITStrack = ITStracks->at(ITSTPCtrack.getRefITS());
                auto &TPCtrack = TPCtracks->at(ITSTPCtrack.getRefTPC());

                std::array<CompClusterExt, 7> TrackClus;
                std::array<ITSCluster, 7> TrackClusXYZ;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &clusXYZ = ITSClusXYZ[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus[layer] = clus;
                    TrackClusXYZ[layer] = clusXYZ;
                }

                for (int layer{0}; layer < 7; layer++)
                {
                    if (ITStrack.hasHitOnLayer(layer))
                    {
                        auto pattID = TrackClus[layer].getPatternID();
                        int npix;
                        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID))
                        {
                            o2::itsmft::ClusterPattern patt(pattIt2);
                            npix = patt.getNPixels();
                        }
                        else
                        {

                            npix = mdict.getNpixels(pattID);
                        }
                        clSizeArr[layer] = npix;
                    }
                    else
                        clSizeArr[layer] = -1;
                }

                if (std::abs(ITSTPCtrack.getEta()) < 0.5)
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

                    hPtRes->Fill((ITStrack.getP() - ITSTPCtrack.getP()) / ITSTPCtrack.getP());
                    hSplines->Fill(ITSTPCtrack.getP(), mean);
                    hSplinesTPC->Fill(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);

                    if (TPCtrack.getP() < 0.9 && TPCtrack.getP() > 0.2 && ITSTPCtrack.getChi2Match() < 20)
                    {

                        p = ITSTPCtrack.getP();
                        pt = ITSTPCtrack.getPt();
                        tgL = ITSTPCtrack.getTgl();
                        meanClsize = mean;
                        dedx = TPCtrack.getdEdx().dEdxTotTPC;

                        double nsigmaDeu = nSigmaDeu(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);

                        double nsigmaP = nSigmaP(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                        double nsigmaPi = nSigmaPi(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);
                        double nsigmaK = nSigmaK(TPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);

                        bool isDeu = nsigmaDeu < 2 && nsigmaPi > 4 && nsigmaK > 4 && nsigmaP > 4;
                        bool isP = nsigmaP < 2 && nsigmaPi > 4 && nsigmaK > 4 && nsigmaDeu > 4;
                        bool isK = nsigmaK < 2 && nsigmaPi > 4 && nsigmaDeu > 4 && nsigmaP > 4;
                        bool isPi = nsigmaPi < 2 && nsigmaDeu > 4 && nsigmaK > 4 && nsigmaP > 4;

                        if (isDeu)
                            label = 3;
                        else if (isP)
                            label = 2;
                        else if (isK)
                            label = 1;
                        else if (isPi)
                            label = 0;
                        else
                            label = -1;

                        if (TPCtrack.getP() > 0.3 && TPCtrack.getP() < 0.4)
                        {
                            if (isDeu)
                            {
                                hClSizeDeu->Fill(mean);
                            }
                            else if (isP)
                            {
                                hClSizeP->Fill(mean);
                            }
                            else if (isK)
                            {
                                hClSizeK->Fill(mean);
                            }
                            else if (isPi)
                            {
                                hClSizePi->Fill(mean);
                            }
                        }

                        for (unsigned int layer{0}; layer < clSizeArr.size(); layer++)
                        {
                            if (ITStrack.hasHitOnLayer(layer))
                            {
                                auto &clusXYZ = TrackClusXYZ[layer];
                                if (propagateToClus(clusXYZ, ITSTPCtrack, gman))
                                    snPhiArr[layer] = ITSTPCtrack.getSnp();
                                else
                                    snPhiArr[layer] = ITSTPCtrack.getSnp();
                            }
                            else
                                snPhiArr[layer] = -2;
                        }
                        MLtree->Fill();
                    }
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

    hSplines->Write();

    int size = 2000;
    Double_t x[size], ex[size], yD[size], yP[size], yPi[size], yK[size], yDerr[size], yPerr[size], yPierr[size], yKerr[size];
    Int_t n = size;

    for (Int_t i = 1; i < n; i++)
    {
        x[i] = i * 0.001;
        yPi[i] = BetheBlochParam(x[i], TDatabasePDG::Instance()->GetParticle(211)->Mass());
        yK[i] = BetheBlochParam(x[i], TDatabasePDG::Instance()->GetParticle(321)->Mass());
        yP[i] = BetheBlochParam(x[i], TDatabasePDG::Instance()->GetParticle(2212)->Mass());
        yD[i] = BetheBlochParam(x[i], 1.87561);

        yPierr[i] = yPi[i] * 0.07;
        yKerr[i] = yK[i] * 0.07;
        yPerr[i] = yP[i] * 0.07;
        yDerr[i] = yD[i] * 0.07;

        ex[i] = 0.;
    }
    TGraphErrors *grPi = new TGraphErrors(n, x, yPi, ex, yPierr);
    TGraphErrors *grK = new TGraphErrors(n, x, yK, ex, yKerr);
    TGraphErrors *grP = new TGraphErrors(n, x, yP, ex, yPerr);
    TGraphErrors *grD = new TGraphErrors(n, x, yD, ex, yDerr);

    grPi->SetLineColor(kRed);
    grK->SetLineColor(kRed);
    grP->SetLineColor(kRed);
    grD->SetLineColor(kRed);

    grPi->SetMarkerColor(kRed);
    grK->SetMarkerColor(kRed);
    grP->SetMarkerColor(kRed);
    grD->SetMarkerColor(kRed);

    grPi->SetFillColor(kRed);
    // grPi->SetFillStyle(3005);
    grP->SetFillColor(kRed);
    // grP->SetFillStyle(3005);
    grK->SetFillColor(kRed);
    grD->SetFillColor(kRed);

    TMultiGraph *mg = new TMultiGraph();
    mg->Add(grPi);
    mg->Add(grP);
    mg->Add(grK);
    mg->Add(grD);

    auto cv = TCanvas("TPC splines", "TPC splines");
    hSplinesTPC->Draw("colz");
    mg->GetXaxis()->SetLimits(0.03, 2);
    mg->Draw("C3");
    cv.Write();

    TCanvas cClSize = TCanvas("Cl size for p and #pi", "Cl size for p and #pi");
    auto leg = new TLegend(0.6, 0.65, 0.9, 0.85);
    hClSizePi->SetLineWidth(2);
    hClSizePi->SetStats(0);
    hClSizePi->DrawNormalized();
    hClSizeK->SetLineColor(kOrange + 2);
    hClSizeK->SetLineWidth(2);
    hClSizeK->DrawNormalized("same");
    hClSizeP->SetLineColor(kRed + 2);
    hClSizeP->SetLineWidth(2);
    hClSizeP->DrawNormalized("same");
    hClSizeDeu->SetLineColor(kGreen + 2);
    hClSizeDeu->SetLineWidth(2);
    // hClSizeDeu->DrawNormalized("same");

    leg->SetHeader("ITS2 #LT Cluster Size #GT, 0.3 < #it{p}^{ITS-TPC} < 0.4 (GeV/#it{c})");
    leg->SetNColumns(2);
    leg->SetMargin(0.1);
    // leg->SetTextSize(1);
    leg->AddEntry(hClSizePi, "#pi", "l");
    leg->AddEntry(hClSizeK, "K", "l");
    leg->AddEntry(hClSizeP, "p", "l");
    // leg->AddEntry(hClSizeDeu, "d", "l");


    leg->Draw();
    cClSize.Write();

    hPtRes->Write();
    hChi2ClSize->Write();

    MLtree->Write();
    outFile.Close();
}

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman, float Bz)
{
    float alpha = gman->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
    if (track.rotate(alpha))
    {
        auto corrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;
        auto propInstance = o2::base::Propagator::Instance();
        float alpha = gman->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
        int layer{gman->getLayer(clus.getSensorID())};

        if (!track.rotate(alpha))
            return false;

        if (propInstance->propagateToX(track, x, Bz, o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, corrType))

            return true;
    }
    return false;
}

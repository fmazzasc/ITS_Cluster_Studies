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
#include <TSystemDirectory.h>
#include <TSystemFile.h>

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

bool propagateToClus(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman, float Bz = -5.);

void pidITS()
{

    TH2D *hSplines = new TH2D("ITS splines ", ";#it{p}^{ITS-TPC} (GeV/#it{c}); #LT Cluster size #GT #times Cos(#lambda) ; Counts", 300, 0, 2, 60, 0.5, 12.5);
    TH2D *hSplinesSA = new TH2D("ITS splines ITS SA ", ";#it{p}^ITS (GeV/#it{c}); #LT Cluster size #GT #times Cos(#lambda) ; Counts", 300, 0, 2, 60, 0.5, 12.5);
    TH1D *hClSizeP = new TH1D("Average Cl size for protons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);
    TH1D *hClSizePi = new TH1D("Average Cl size for pi", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 40, 0.5, 12.5);
    TH1D *hTotClSizeP = new TH1D("Cl size for protons ", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 13, 0.5, 13.5);
    TH1D *hTotClSizePi = new TH1D("Cl size for pi", "; #LT Cluster size #GT #times Cos(#lambda) ; Normalised Counts", 13, 0.5, 13.5);
    TH2D *hSplinesTPC = new TH2D("TPC splines ", ";#it{p}^{ITS-TPC} (GeV/#it{c}); TPC Signal ; Counts", 300, 0.05, 2, 300, 30.5, 600.5);
    TH1D *hPtRes = new TH1D("pT resolution ", ";(#it{p}^{ITS-SA} - #it{p}^{ITS-TPC})/#it{p}^{ITS-TPC}; Counts", 80, -1, 1);

    TFile outFile = TFile("pid.root", "recreate");

    TTree *MLtree = new TTree("ITStreeML", "ITStreeML");
    std::array<float, 19> cand;
    MLtree->Branch("TrackInfo", &cand);

    bool usePaki = false;

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2");
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
        // if (counter > 50)
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

                std::vector<CompClusterExt> TrackClus;
                std::vector<ITSCluster> TrackClusXYZ;

                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &clusXYZ = ITSClusXYZ[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto layer = gman->getLayer(clus.getSensorID());
                    TrackClus.push_back(clus);
                    TrackClusXYZ.push_back(clusXYZ);
                }

                std::reverse(TrackClus.begin(), TrackClus.end());
                std::reverse(TrackClusXYZ.begin(), TrackClusXYZ.end());

                std::vector<int> clusterSizes;

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
                        clusterSizes.push_back(npix);
                    }
                }


                std::sort(clusterSizes.begin(), clusterSizes.end());
                float mean = 0;
                if (usePaki)
                {
                    for (unsigned int i{0}; i < clusterSizes.size(); i++)
                    {
                        mean += clusterSizes[i] * pakiWeights[6 - i];
                    }
                    mean /= sumPaki;
                }
                else
                {
                    for (unsigned int i{0}; i < clusterSizes.size(); i++)
                    {
                        mean += clusterSizes[i];
                    }
                    mean /= (clusterSizes.size());
                }
                mean *= std::sqrt(1. / (1 + ITSTPCtrack.getTgl() * ITSTPCtrack.getTgl()));
                if ((clusterSizes.size() == 7 && usePaki == false) && std::abs(ITSTPCtrack.getEta()) < 0.5)
                {

                    hPtRes->Fill((ITStrack.getP() - ITSTPCtrack.getP()) / ITSTPCtrack.getP());
                    hSplines->Fill(ITSTPCtrack.getP(), mean);
                    hSplinesTPC->Fill(ITSTPCtrack.getP(), TPCtrack.getdEdx().dEdxTotTPC);

                    if (ITSTPCtrack.getP() > 0.3 && ITSTPCtrack.getP() < 0.4)
                    {
                        if (TPCtrack.getdEdx().dEdxTotTPC > 240)
                        {
                            hClSizeP->Fill(mean);
                            cand[18] = 1.;
                        }
                        if (TPCtrack.getdEdx().dEdxTotTPC < 80)
                        {
                            hClSizePi->Fill(mean);
                            cand[18] = 0.;
                        }

                        cand[14] = ITSTPCtrack.getP();
                        cand[15] = ITSTPCtrack.getPt();

                        cand[16] = ITSTPCtrack.getPhi();
                        cand[17] = mean;

                        for (unsigned int i{0}; i < clusterSizes.size(); i++)
                        {
                            auto &clusXYZ = TrackClusXYZ[i];
                            auto &iSize = clusterSizes[i];
                            cand[i] = iSize;

                            if (propagateToClus(clusXYZ, ITSTPCtrack, gman))
                                cand[i + 7] = ITSTPCtrack.getTgl();
                            else
                                cand[i + 7] = 0;

                            if (TPCtrack.getdEdx().dEdxTotTPC > 240)
                            {
                                hTotClSizeP->Fill(iSize);
                            }
                            if (TPCtrack.getdEdx().dEdxTotTPC < 80)
                            {
                                hTotClSizePi->Fill(iSize);
                            }
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
    hSplinesTPC->Write();

    TCanvas cClSize = TCanvas("Cl size for p and #pi", "Cl size for p and #pi");
    auto leg = new TLegend(0.6, 0.65, 0.9, 0.85);
    hClSizePi->SetLineWidth(2);
    hClSizePi->SetStats(0);
    hClSizePi->Draw();
    hClSizeP->SetLineColor(kRed + 2);
    hClSizeP->SetLineWidth(2);
    hClSizeP->Draw("same");
    leg->SetHeader("ITS2 #LT Cluster Size #GT, 0.3 < #it{p}^{ITS-TPC} < 0.4 (GeV/#it{c})");
    leg->SetMargin(0.1);
    leg->SetTextSize(2);
    leg->AddEntry(hClSizePi, "#pi", "l");
    leg->AddEntry(hClSizeP, "p", "l");
    leg->Draw();
    cClSize.Write();

    TCanvas cClSizeTot = TCanvas("Total Cl size for p and #pi", "Cl size for p and #pi");
    auto leg2 = new TLegend(0.6, 0.65, 0.9, 0.85);
    hTotClSizePi->SetLineWidth(2);
    hTotClSizePi->SetStats(0);
    hTotClSizePi->DrawNormalized();
    hTotClSizeP->SetLineColor(kRed + 2);
    hTotClSizeP->SetLineWidth(2);
    hTotClSizeP->DrawNormalized("same");
    leg2->SetHeader("ITS2 #LT Cluster Size #GT, 0.3 < #it{p}^{ITS-TPC} < 0.4 (GeV/#it{c})");
    leg2->SetMargin(0.1);
    leg2->SetTextSize(2);
    leg2->AddEntry(hTotClSizePi, "#pi", "l");
    leg2->AddEntry(hTotClSizeP, "p", "l");
    leg2->Draw();

    cClSizeTot.Write();
    hPtRes->Write();

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

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
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

void checkTrackLayerCorrelation()
{

    int npixThr = 10;
    TH1D *hClSizeAll = new TH1D("hClSizeAll", "hClSizeAll", 50, 0.5, 50.5);
    std::vector<TH1D *> hClSizeAllvsLayer(7);
    TH1D *hClSizeFilt = new TH1D("hClSizeFilt", "hClSizeFilt", 50, 0.5, 50.5);
    std::vector<TH1D *> hClSizeFiltvsLayer(7);

    for (int layer{0}; layer < 7; layer++)
    {
        hClSizeAllvsLayer[layer] = new TH1D(Form("hClSizeAllvsLayer%i", layer), Form("hClSizeAllvsLayer%i", layer), 100, 0.5, 100.5);
        hClSizeFiltvsLayer[layer] = new TH1D(Form("hClSizeFiltvsLayer%i", layer), Form("hClSizeFiltvsLayer%i", layer), 100, 0.5, 100.5);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("utils/o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "utils/ITSdictionary.bin"));

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

        for (int frame = 0; frame < treeITS->GetEntriesFast(); frame++)
        {
            // LOG(info) << frame;
            // if (frame > 10000)
            //     continue;

            if (!treeITSclus->GetEvent(frame) || !treeITS->GetEvent(frame))
                continue;

            auto pattIt = ITSpatt->cbegin();

            for (unsigned int iTrack{0}; iTrack < ITStracks->size(); ++iTrack)
            {

                auto &ITStrack = ITStracks->at(iTrack);
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
                std::vector<double> clSize = {0., 0., 0., 0., 0., 0., 0.};
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
                        clSize[layer] = npix;
                        // LOG(info) << "CL size " << clSize[layer];
                    }
                    else
                    {
                        clSize[layer] = -1;
                    }
                }

                int maxClpos = std::distance(clSize.begin(), std::max_element(clSize.begin(), clSize.end()));
                for (int icl{0}; icl < 7; icl++) // filling size All
                {
                    if (icl != maxClpos)
                    {
                        hClSizeAll->Fill(clSize[icl]);
                        hClSizeAllvsLayer[icl]->Fill(clSize[icl]);
                    }

                    else
                    {
                        if (clSize[icl] > npixThr)
                        {
                            for (int jcl{0}; jcl < 7; jcl++)
                            {
                                // LOG(info) << "iTrack" << iTrack << " CL size accepted " << j ;
                                if (jcl != maxClpos)
                                {
                                    hClSizeFilt->Fill(clSize[jcl]);
                                    hClSizeFiltvsLayer[jcl]->Fill(clSize[jcl]);
                                }
                            }
                        }
                    }
                }
            }
        }
        treeITS->ResetBranchAddresses();
        treeITSclus->ResetBranchAddresses();
        fITS->Close();
        fITSTPC->Close();
        fITSclus->Close();
    }
    LOG(info) << "Saving output";

    auto outFile = TFile(Form("clusITStrackCorr%i.root", npixThr), "recreate");
    TCanvas clusITStrackCorr = TCanvas("clusITStrackCorr", "clusITStrackCorr", 800, 800);
    // std::vector<TCanvas *> clusITStrackCorrVsLayer(7);
    clusITStrackCorr.Divide(2);
    clusITStrackCorr.cd(1)->SetLogy();
    hClSizeAll->Scale(1. / hClSizeAll->Integral());
    // hClSizeAll->SetFillStyle(3004);
    hClSizeAll->Draw();
    hClSizeFilt->Scale(1. / hClSizeFilt->Integral());
    // hClSizeFilt->SetFillStyle(3005);
    hClSizeFilt->SetLineColor(kRed);
    hClSizeFilt->Draw("same");
    TH1D *hClSizeRatio = (TH1D *)hClSizeFilt->Clone("hClSizeRatio");
    hClSizeRatio->Divide(hClSizeAll);
    clusITStrackCorr.cd(2);
    hClSizeRatio->Draw();

    for (int layer{0}; layer < 7; layer++)
    {
        TCanvas cClusterSize = TCanvas(Form("cClusterSizeL%i", layer), Form("cClusterSizeL%i", layer));
        cClusterSize.SetLogy();
        auto leg2 = new TLegend(0.33, 0.65, 0.95, 0.85);
        leg2->SetMargin(0.1);
        leg2->SetHeader(Form("ALICE pp #sqrt{s} = 900 GeV, ITS2 Layer %i", layer));
        hClSizeAllvsLayer[layer]->SetStats(0);
        hClSizeAllvsLayer[layer]->DrawNormalized("HIST");
        hClSizeFiltvsLayer[layer]->SetStats(0);
        hClSizeFiltvsLayer[layer]->SetLineColor(kRed + 2);
        hClSizeAllvsLayer[layer]->SetLineWidth(2);
        hClSizeFiltvsLayer[layer]->SetLineWidth(2);
        hClSizeFiltvsLayer[layer]->DrawNormalized("HIST SAME");
        leg2->AddEntry(hClSizeAllvsLayer[layer], "All clusters", "l");
        leg2->AddEntry(hClSizeFiltvsLayer[layer], "Correlated only", "l");
        leg2->Draw();
        cClusterSize.Write();

        // hClSizeAllvsLayer[layer]->Write();
        // hClSizeFiltvsLayer[layer]->Write();
        // clusITStrackCorrVsLayer[layer]->Write();
    }
    hClSizeAll->Write();
    hClSizeFilt->Write();
    clusITStrackCorr.Write();
    outFile.Close();
}
#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "../utils/ClusterStudyUtils.h"

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCTrack.h"
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
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"

#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using MCTrack = o2::MCTrack;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using CompClusterExt = o2::itsmft::CompClusterExt;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void mcOrigin(std::string inPath="", std::string outLabel="", bool isOldData=false, unsigned int pix_thr = 40, bool verbose=false) 
{
    /*
     - outLabel: label added to final root file 
     - isOldData: bool to adopt old/new dictionary
     - verbose: allow additional print
    */
    // "------------------ GLOBAL info ------------------"
    o2::base::GeometryManager::loadGeometry("../utils/o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    std::vector<int> nStaves{12, 16, 20, 24, 30, 42, 48};
    std::vector<double> deltaEta{0.3, 0.44, 0.6, 0.72, 0.77, 0.72, 0.6, 0.44, 0.3};

    TH1D *hCLid = new TH1D("hCLid", ";MC Track ID; entries", 8000, -0.5, 7999.5);
    TH1D *hTrackCLid = new TH1D("hTrackCLid", ";MC Track ID; entries", 10000, -0.5, 9999.5);
    TH1D *hPDGcodeCL = new TH1D("hPDGcodeCL", ";MC Track ID; entries", 10000, -0.5, 9999.5);
    TH1D *hPDGmass = new TH1D("hPDGmass", ";PDG mass (); entries", 10000, -0.5, 99.5);
    TH1D *hPDGeloss = new TH1D("hPDGeloss", ";PDG dE/dx (); entries", 10000, -0.5, 99.5);
    TH1D *hPDGp = new TH1D("hPDGp", ";PDG p (); entries", 50000, -0.5, 49.5);
    hCLid->SetDirectory(nullptr);
    hTrackCLid->SetDirectory(nullptr);
    hPDGmass->SetDirectory(nullptr);
    hPDGeloss->SetDirectory(nullptr);
    hPDGp->SetDirectory(nullptr);


    LOG(info) << "------------------ LOADING INPUT FILES ------------------";
    // Topology dictionary
    if (verbose){
        LOG(info) << "Loading topology dictionary";
    }
    o2::itsmft::TopologyDictionary mdict;
    o2::itsmft::ChipMappingITS chipMapping;
    if (isOldData)
    {
        LOG(info) << "Loading OLD dictionary: if you are analysing data older than JUNE should be fine";
        mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "../utils/ITS"));
    }
    else
    {
        LOG(info) << "Loading LATEST dictionary: if you are analysing data older than JUNE check out the dictionary";
        auto f = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
        mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(f.Get("ccdb_object")));
    }

    // Define the PB input file
    if (verbose)
    {
        LOG(info) << "Loading PB data file from " << inPath;
    }
    TSystemDirectory dir("MyDir", inPath.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs;
    for (auto fileObj : *files)
    {
        std::string file = ((TSystemFile *)fileObj)->GetName();
        if (verbose)
        {
            LOG(info) << "Keeping " << file;
        } 
        dirs.push_back(inPath+file);
    }
    std::sort(dirs.begin(), dirs.end());

    std::vector<std::string> fulldirs;
    for (auto &dir : dirs)
    {
        TSystemDirectory subdir("MyDir2", dir.data());  
        auto files = subdir.GetListOfFiles(); 
        for (auto fileObj : *files)
        {
            std::string file = ((TSystemFile *)fileObj)->GetName();
            if (file.substr(0, 2) == "tf")
            {
                if (verbose)
                {
                    LOG(info) << "Keeping " << file;
                } 
                fulldirs.push_back(dir + "/" + file);
            }
        }
    }
    std::sort(fulldirs.begin(), fulldirs.end());

    int counter = 0;
    for (auto &dir : fulldirs)
    {   
        if (verbose)
        {
            if (counter > 20)
            {
                continue;
            }
            counter ++;
        }
        LOG(info) << "Analysing directory: " << dir;
        std::string o2clus_its_file = dir + "/" + "o2clus_its.root";
        std::string o2trac_its_file = dir + "/" + "o2trac_its.root";
        std::string o2kine_file = dir + "/" + Form("sgn_%s_Kine.root", dir.substr(37, dir.size()).data());

        auto fITSclus = TFile(o2clus_its_file.data());
        auto treeITSclus = (TTree *)fITSclus.Get("o2sim");
        auto fITStrac = TFile::Open(o2trac_its_file.data());
        auto treeITStrac = (TTree *)fITStrac->Get("o2sim");
        auto fMCTracks = TFile(o2kine_file.data());
        auto treeMCTracks = (TTree *)fMCTracks.Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;
        o2::dataformats::MCTruthContainer<o2::MCCompLabel> *clusLabArr = nullptr;
        std::vector<int> *ITSTrackClusIdx = nullptr;
        std::vector<o2::its::TrackITS> *ITStracks = nullptr;
        std::vector<o2::MCTrack> *MCtracks = nullptr;
        
        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
        treeITSclus->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
        treeITStrac->SetBranchAddress("ITSTrack", &ITStracks);
        treeITStrac->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);
        treeMCTracks->SetBranchAddress("MCTrack", &MCtracks);
        
        std::vector<int> LargeCLTrackID;
        std::vector<int> LargeCLEvID;
        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {   // LOOP OVER FRAMES  
            if (!treeITSclus->GetEvent(frame) || !treeITStrac->GetEvent(frame))
            {
                if(verbose)
                {
                    LOG(info) << "Skipping frame: " << frame;
                }
                continue;
            }

            if(verbose) 
            {
                LOG(info) << "Frame: " << frame;
            }
            std::vector<o2::itsmft::ClusterPattern> pattVec;
            getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);
            for (unsigned int iClus{0}; iClus < ITSclus->size(); iClus++)
            {   // LOOP OVER CLUSTERS
                auto &patt = pattVec[iClus];
                auto &clus = ITSclus->at(iClus);
                auto chipID = clus.getChipID();
                int layer, sta, ssta, mod, chipInMod;
                auto pattID = clus.getPatternID();
                int npix = patt.getNPixels();

                if (npix > pix_thr)
                {
                    auto &labCls = (clusLabArr->getLabels(iClus))[0];
                    int  trackID, evID, srcID;
                    bool fake;
                    labCls.get(trackID, evID, srcID, fake);
                    if (verbose)
                    {
                        LOG(info) << "Labels info: trackID="<<trackID<<", eventID="<<evID<<", srcID="<<srcID;
                    }
                    LargeCLTrackID.push_back(trackID);
                    LargeCLEvID.push_back(evID);
                    hCLid->Fill(trackID);
                }

            }
        }
            /*
            o2::its::TrackITS ITStrack;
            for (unsigned int iTrack{0}; iTrack < ITStracks->size(); iTrack++)
            {   // LOOP OVER TRACKS
                if (iTrack%10 == 0 && verbose)
                {
                    LOG(info) << "iTrack: " << iTrack;
                }

                auto &patt = pattVec[iTrack];
                ITStrack = (*ITStracks)[iTrack];
                auto firstClus = ITStrack.getFirstClusterEntry();
                auto ncl = ITStrack.getNumberOfClusters();

                for (int icl = 0; icl < ncl; icl++)
                {   // LOOP OVER CLUSTERS
                    auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
                    auto &patt = pattVec[(*ITSTrackClusIdx)[firstClus + icl]];
                    
                    int npix = patt.getNPixels();
                    if (npix > pix_thr)
                    {
                        auto &labCls = (clusLabArr->getLabels(ITSTrackClusIdx->at(firstClus+icl)))[0];
                        int  trackID, evID, srcID;
                        bool fake;
                        labCls.get(trackID, evID, srcID, fake);
                        if (verbose)
                        {
                            LOG(info) << "Labels info: trackID="<<trackID<<", eventID="<<evID<<", srcID="<<srcID;
                        }
                        hTrackCLid->Fill(trackID);
                    }
                }
            }
            */

        std::vector<std::vector<o2::MCTrack>> mcTracksMatrix;
        auto nev = treeMCTracks->GetEntriesFast();
        mcTracksMatrix.resize(nev);
        for (int n = 0; n < nev; n++)
        { // loop over MC events
            treeMCTracks->GetEvent(n);
            mcTracksMatrix[n].resize(MCtracks->size());
            if (verbose)
            {
                LOG(info) << "N MC ev.=" << nev <<", N MC tracks="<<MCtracks->size();
            }
            for (unsigned int mcI{0}; mcI < MCtracks->size(); ++mcI)
            {   // LOOP over MC tracks
                mcTracksMatrix[n][mcI] = MCtracks->at(mcI);
            }
        }

        LOG(info) << "---- GETTING MC tracks info ----";
        for (int i = 0; i < LargeCLEvID.size(); i++)
        {
            auto evID = LargeCLEvID.at(i);
            auto trID = LargeCLTrackID.at(i);

            if (verbose)
            {
                LOG(info) <<"evID="<<evID<<", trID="<<trID;
            }
            auto trPDG = mcTracksMatrix[evID][trID].GetPdgCode();
            double mass = TDatabasePDG::Instance()->GetParticle(trPDG)->Mass();
            double p = mcTracksMatrix[evID][trID].GetP();
            if (verbose)
            {
                LOG(info) <<"m="<<mass<<", p="<<p<<", PDG="<<trPDG;
            }
            hPDGmass->Fill(mass);
            hPDGp->Fill(p);
        }
    }

    LOG(info) << "------------------ SAVING OUTFILE ------------------";
    auto outFile = TFile(Form("outFileMCid_%s.root", outLabel.data()), "recreate");
    hCLid->Write();
    hTrackCLid->Write();
    hPDGmass->Write();
    hPDGp->Write();
}

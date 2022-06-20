#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "../utils/ClusterStudyUtils.h"

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
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using CompClusterExt = o2::itsmft::CompClusterExt;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;

void pbdataClSizeAnalyser(bool verbose=false) 
{
    // Geometry
    o2::base::GeometryManager::loadGeometry("../utils/o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));


    LOG(INFO) << "------------------ PB DATA FROM JUNE ------------------";
    // Topology dictionary
    if (verbose){
        LOG(INFO) << "Loading topology dictionary";
    }
    o2::itsmft::TopologyDictionary mdict;
    auto f = TFile("../utils/o2_itsmft_TopologyDictionary_1653153873993.root");
    mdict = *(reinterpret_cast<o2::itsmft::TopologyDictionary *>(f.Get("ccdb_object")));

    // Define the PB input file
    if (verbose)
    {
        LOG(INFO) << "Loading PB data file from o2clus_its_June.root";
    }
    std::string o2clus_its_file = "o2clus_its_June.root";
    auto fITSclus = TFile(o2clus_its_file.data());
    auto treeITSclus = (TTree *)fITSclus.Get("o2sim");
    std::vector<CompClusterExt> *ITSclus = nullptr;
    std::vector<unsigned char> *ITSpatt = nullptr;
    treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
    treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

    // Define the output file
    if (verbose)
    {
        LOG(INFO) << "Preparing outfile: outFileClSizeJuneData.root";
    }
    auto outFile = TFile::Open("outFileClSizeJuneData.root", "recreate");
    TH1D *hClSize = new TH1D("hClSizehJune", ";Cluster size; entries", 100, 0.5, 100.5);

    // Filling histos
    if (verbose)
    {
        LOG(INFO) << "Filling histogram for JUNE";
    }
    for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
    {
        if (!treeITSclus->GetEvent(frame))
            continue;
        std::vector<o2::itsmft::ClusterPattern> pattVec;
        getClusterPatterns(pattVec, ITSclus, ITSpatt, mdict, gman);
        for (unsigned int iClus{0}; iClus < ITSclus->size(); iClus++)
        {
            if (iClus%10 == 0)
            {
                LOG(INFO) << iClus;
            }
            auto &patt = pattVec[iClus];
            auto &clus = ITSclus->at(iClus);
            auto chipID = clus.getChipID();
            int layer, sta, ssta, mod, chipInMod;
            auto pattID = clus.getPatternID();
            int npix = patt.getNPixels();
            hClSize->Fill(npix);
            //LOG(INFO) << "Filling histogram for JUNE" << hClSize->GetEntries();
        }
    }
    
    hClSize->Write();
    hClSize->SaveAs("hClSizeJune.root");

    LOG(INFO) << "------------------ PB DATA FROM OCT ------------------";

    // Define the output file
    TH1D *hClSizeOct = new TH1D("hClSizehOct", ";Cluster size; entries", 100, 0.5, 100.5);
    
    // Topology dictionary
    if (verbose){
        LOG(INFO) << "Loading topology dictionary";
    }
    std::string path = "/data/fmazzasc/its_data/PBdata/BPOS/505658";
    o2::itsmft::TopologyDictionary mdictOct;
    mdictOct.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "../utils/ITS"));
    TSystemDirectory dir("MyDir", path.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs;
    for (auto fileObj : *files)
    {
        std::string file = ((TSystemFile *)fileObj)->GetName();
        if (file.substr(0, 6) == "o2_ctf" || file.substr(0, 2) == "tf")
            dirs.push_back(file);
    }
    int counter = 0;
    for (auto &dir : dirs)
    {

        LOG(info) << "Processing: " << counter << ", dir: " << dir;
        if (counter > 100)
             continue;
        counter++;
        std::string o2clus_its_file;
        std::string primary_vertex_file;

        o2clus_its_file = path + "/" + dir + "/" + "o2clus_its.root";
        auto fITSclusOct = TFile::Open(o2clus_its_file.data());
        if (!fITSclusOct)
            continue;

        auto treeITSclus = (TTree *)fITSclusOct->Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);



        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {
            if (!treeITSclus->GetEvent(frame))
                continue;
            std::vector<o2::itsmft::ClusterPattern> pattVecOct;
            getClusterPatterns(pattVecOct, ITSclus, ITSpatt, mdictOct, gman);
            for (unsigned int iClus{0}; iClus < ITSclus->size(); iClus++)
            {
                auto &patt = pattVecOct[iClus];
                auto &clus = ITSclus->at(iClus);
                auto chipID = clus.getChipID();
                int layer, sta, ssta, mod, chipInMod;
                auto pattID = clus.getPatternID();
                int npix = patt.getNPixels();
                hClSizeOct->Fill(npix);
            }
        }
    }
    hClSizeOct->SaveAs("hClSizeOct.root");
}
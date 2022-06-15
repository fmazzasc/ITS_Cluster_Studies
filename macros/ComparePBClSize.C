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

void pbdataClSizeAnalyser() 
{
    // Geometry
    o2::base::GeometryManager::loadGeometry("../utils/o2_geometry.root");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

    // Topology dictionary
    LOG(INFO) << "Loading topology dictionary";
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "../utils/ITS"));


    // Define the input file
    std::string o2clus_its_file = "o2clus_its_June.root";
    auto fITSclus = TFile(o2clus_its_file.data());
    auto treeITSclus = (TTree *)fITSclus.Get("o2sim");

    std::vector<CompClusterExt> *ITSclus = nullptr;
    std::vector<unsigned char> *ITSpatt = nullptr;
    treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
    treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

    // Define the output file
    auto outFile = TFile::Open("outFileClSizeJuneData.root", "recreate");
    TH1D *hClSize = new TH1D("hClSize", ";Cluster size; Norm. entries", 100, 0.5, 100.5);


    // Filling the histogram
    std::vector<o2::itsmft::ClusterPattern> pattVec;
    pattVec.reserve(ITSclus->size());
    auto pattIt = ITSpatt->cbegin();
    LOG(INFO) << "ITSclus->size() = " << ITSclus->size();
    for (unsigned int iClus{0}; iClus < ITSclus->size(); iClus++) {
        //LOG(INFO) << "Cluster " << iClus << " size " << ITSclus->size();
        auto &clus = (*ITSclus)[iClus];
        o2::itsmft::ClusterPattern patt;
        auto pattID = clus.getPatternID();
        int npix;
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID))
        {
            //LOG(INFO) << "Cluster " << iClus << " has invalid pattern ID " << pattID;
            patt.acquirePattern(pattIt);
            npix = patt.getNPixels();
        }
        else
        {
            //LOG(INFO) << "Cluster " << iClus << " has valid pattern ID " << pattID;
            npix = mdict.getNpixels(pattID);
            patt = mdict.getPattern(pattID);
        }
        //LOG(INFO) << "Cluster  with " << npix << " pixels";
        hClSize->Fill(npix);
    }
}
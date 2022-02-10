#if !defined(CLING) || defined(ROOTCLING)

#include <iostream>

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#include "ITSMFTReconstruction/ChipMappingITS.h"

#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include <TRandom.h>
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
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"

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

void averageClSize()
{
    gStyle->SetPalette(82);
    gStyle->SetPadRightMargin(0.035);
    gStyle->SetPadLeftMargin(0.005);

    o2::itsmft::ChipMappingITS chipMapping;
    std::vector<TH1D *> AverageClSize(7);

    std::vector<int> runNumbers = {505548, 505582, 505600, 505645, 505658, 505629};

    for (int layer{0}; layer < 7; layer++)
    {
        AverageClSize[layer] = new TH1D(Form("Average Cluster Size L%i", layer), Form("; ;<Cluster size for L%i>", layer), runNumbers.size(), 0, runNumbers.size() - 1);
        // AverageClSize[layer]->SetLineStyle(5);
        AverageClSize[layer]->SetLineColor(kRed + 2);
    }

    // Geometry
    o2::base::GeometryManager::loadGeometry("o2");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
    // Topology dictionary
    o2::itsmft::TopologyDictionary mdict;
    mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

    for (unsigned int iRun{0}; iRun < runNumbers.size(); iRun++)
    {
        auto &runNum = runNumbers[iRun];
        std::ostringstream strDir;
        strDir << runNum;
        auto dir = strDir.str();
        std::string o2clus_its_file = dir + "/" + "o2clus_its.root";

        auto fITSclus = TFile::Open(o2clus_its_file.data());
        auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

        std::vector<CompClusterExt> *ITSclus = nullptr;
        std::vector<unsigned char> *ITSpatt = nullptr;

        treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
        treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

        std::vector<double> sumVec[7];

        for (int frame = 0; frame < treeITSclus->GetEntriesFast(); frame++)
        {

            if (!treeITSclus->GetEvent(frame))
                continue;

            auto pattIt = ITSpatt->cbegin();

            for (auto &clus : *ITSclus)
            {
                o2::itsmft::ClusterPattern patt;
                auto layer = gman->getLayer(clus.getSensorID());
                auto pattID = clus.getPatternID();
                int npix;
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

                sumVec[layer].push_back(npix);
            }
        }

        for (int layer{0}; layer < 7; layer++)
        {
            auto &v = sumVec[layer];

            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            double mean = sum / v.size();

            double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / v.size() - mean * mean);

            AverageClSize[layer]->SetBinContent(iRun + 1, mean);
            AverageClSize[layer]->SetBinError(iRun + 1, stdev);
            AverageClSize[layer]->GetXaxis()->SetBinLabel(iRun + 1, Form("%i",runNum));
            AverageClSize[layer]->SetStats(0);
        }
    }

    auto outFile = TFile("av_cl_size.root", "recreate");
    for (int layer{0}; layer < 7; layer++)
        AverageClSize[layer]->Write();
    outFile.Close();
}

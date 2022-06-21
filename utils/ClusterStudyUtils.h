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


bool propagateToClusITS(const ITSCluster &clus, o2::track::TrackParCov &track, o2::its::GeometryTGeo *gman)
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

        if (propInstance->propagateToX(track, x, propInstance->getNominalBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, corrType))
            return true;
    }

    return false;
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
            patt.acquirePattern(pattIt);
            npix = patt.getNPixels();
        }
        else
        {

            npix = mdict.getNpixels(pattID);
            patt = mdict.getPattern(pattID);
        }
        // LOG(info) << "npix: " << npix << " Patt Npixel: " << patt.getNPixels();
        pattVec.push_back(patt);
    }
        // LOG(info) << " Patt Npixel: " << pattVec[0].getNPixels();
}


void printClusTrackInfo(const std::array<CompClusterExt, 7> &TrackClus, const std::array<o2::itsmft::ClusterPattern, 7> &TrackPatt, o2::its::TrackITS &ITStrack)
{
    for (int layer{0}; layer < 7; layer++)
    {
        if (ITStrack.hasHitOnLayer(layer))
        {

            LOG(info) << "Layer " << layer << ": " << TrackPatt[layer].getNPixels() << " pixels";
            LOG(info) << "Track P: " << ITStrack.getP() << " Eta: " << ITStrack.getEta();
        }
    }
}


void fillClusterMap(CompClusterExt &clus, o2::itsmft::ClusterPattern &patt, TH2D *histo)
{

    auto col = clus.getCol();
    auto row = clus.getRow();

    // LOG(info) << "row: " << row << "col: " << col;
    // LOG(info) << patt;

    int ic = 0, ir = 0;

    auto colSpan = patt.getColumnSpan();
    auto rowSpan = patt.getRowSpan();
    auto nBits = rowSpan * colSpan;

    for (int i = 2; i < patt.getUsedBytes() + 2; i++)
    {
        unsigned char tempChar = patt.getByte(i);
        int s = 128; // 0b10000000
        while (s > 0)
        {
            if ((tempChar & s) != 0) // checking active pixels
            {
                histo->Fill(col + ic, row + rowSpan - ir);
            }
            ic++;
            s >>= 1;
            if ((ir + 1) * ic == nBits)
            {
                break;
            }
            if (ic == colSpan)
            {
                ic = 0;
                ir++;
            }
            if ((ir + 1) * ic == nBits)
            {
                break;
            }
        }
    }
}

void fillIBmap(TH2D *histo, CompClusterExt &clus, o2::itsmft::ChipMappingITS &chipMapping, int weight)
{
    auto chipID = clus.getChipID();
    int lay, sta, ssta, mod, chipInMod;
    chipMapping.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);
    histo->Fill(chipInMod, sta, weight);
    // if((chipInMod) == 8 && lay==0) LOG(info) << weight;
}


static bool npix_compare(o2::itsmft::ClusterPattern a, o2::itsmft::ClusterPattern b)
{
    return std::abs(a.getNPixels()) < std::abs(b.getNPixels());
}

void TrackLayerCorr(o2::its::TrackITS &ITStrack, const std::array<CompClusterExt, 7>clus, const std::array<o2::itsmft::ClusterPattern, 7>patt, TH1D* hClSizeAll, std::vector<TH1D *> hClSizeAllvsLayer, int pixThr, std::string cut)
{
    int maxClpos = std::distance(patt.begin(), std::max_element(patt.begin(), patt.end(), npix_compare));
    int npixTrigger = patt[maxClpos].getNPixels();
    
    //LOG(info) << "maxClpos: " << maxClpos;

    if ((cut == "upper") && (npixTrigger > pixThr))
    { 
        for (int layer{0}; layer < 7; layer++) // loop over layer
        {
            //LOG(info) << "Layer: " << layer;
            if (layer != maxClpos)
            {
                if (ITStrack.hasHitOnLayer(layer)) // check hit on layer
                {
                    auto &pattern = patt[layer];
                    auto npix = pattern.getNPixels();
                    //LOG(info) << "npix: " << npix;
                    hClSizeAll->Fill(npix);
                    hClSizeAllvsLayer[layer]->Fill(npix);
                }
            }
        }
    }
    else if ((cut == "lower") && (npixTrigger <= pixThr)) // npix <= 50
    {
        for (int layer{0}; layer < 7; layer++) // loop over layer
        {
            //LOG(info) << "Layer: " << layer;
            if (layer != maxClpos)
            {
                if (ITStrack.hasHitOnLayer(layer)) // check hit on layer
                {
                    auto &pattern = patt[layer];
                    auto npix = pattern.getNPixels();
                    //LOG(info) << "npix: " << npix;
                    hClSizeAll->Fill(npix);
                    hClSizeAllvsLayer[layer]->Fill(npix);
                }
            }
        }
    }
    else if ((cut == "all")) // exclude only trigger particle
    {
        for (int layer{0}; layer < 7; layer++) // loop over layer
        {
            //LOG(info) << "Layer: " << layer;
            if (layer != maxClpos)
            {
                if (ITStrack.hasHitOnLayer(layer)) // check hit on layer
                {
                    auto &pattern = patt[layer];
                    auto npix = pattern.getNPixels();
                    //LOG(info) << "npix: " << npix;
                    hClSizeAll->Fill(npix);
                    hClSizeAllvsLayer[layer]->Fill(npix);
                }
            }
        }
    }
}

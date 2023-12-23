
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
#include "ReconstructionDataFormats/PrimaryVertex.h"
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

void EventCounterData(int runNumber = 520147, std::string runPeriod = "JUL")
{

    int nEvents = 0;
    std::string pathDir;

    if (runPeriod == "OCT")
    {
        pathDir = "/data/shared/ITS/OCT/CTFS";
    }
    else if (runPeriod == "MAY")
    {
        pathDir = "/data/shared/ITS/MAY/CTFS";
    }
    else if (runPeriod == "JUN")
    {
        pathDir = "/data/shared/ITS/JUN/CTFS";
    }

    else if (runPeriod == "JUL")
    {
        pathDir = "/data/shared/ITS/JUL/pp13TeV/apass/LHC22f";
    }

    else
    {
        LOG(fatal) << "Run period not recognized";
    }

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

    // std::sort(dirs.begin(), dirs.end());

    int counter = 0;

    for (auto iDir{0}; iDir < dirs.size(); iDir++)
    {
        auto dir = dirs[iDir];

        // if (counter > 50)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << " / " << dirs.size() << ", " << dir;
        std::string pv_file = path + "/" + dir + "/" + "root_archive.zip#o2_primary_vertex.root";
        // Files
        auto fPV = TFile::Open(pv_file.data());
        std::vector<o2::dataformats::PrimaryVertex> *PVs = nullptr;

        if (!fPV)
            continue;
        auto treePV = (TTree *)fPV->Get("o2sim");
        // Setting branches
        treePV->SetBranchAddress("PrimaryVertex", &PVs);

        for (int frame = 0; frame < treePV->GetEntriesFast(); frame++)
        {

            if (!treePV->GetEvent(frame))
                continue;

            // creating MC matrix
            auto nev = PVs->size();
            nEvents += nev;
        }
    }
    LOG(info) << "Number of events: " << nEvents;
}

void EventCounterMC(int runNumber = 505548, std::string runPeriod = "OCT")
{

    int nEvents = 0;
    std::string pathDir;

    if (runPeriod == "OCT")
    {
        pathDir = "/data/shared/ITS/OCT/MC";
    }

    else
    {
        LOG(fatal) << "Run period not recognized";
    }

    std::string path = Form("%s/%i", pathDir.data(), runNumber);
    LOG(info) << "Reading from " << path;
    TSystemDirectory dir("MyDir", path.data());
    auto files = dir.GetListOfFiles();
    std::vector<std::string> dirs, kineNames;
    for (auto fileObj : *files)
    {
        std::string dirStr = ((TSystemFile *)fileObj)->GetName();
        TSystemDirectory innerDir("InnerDir", (path + "/" + dirStr).data());
        auto innerFiles = innerDir.GetListOfFiles();
        for (auto innerFile : *innerFiles)
        {
            std::string innerDirStr = ((TSystemFile *)innerFile)->GetName();
            if (innerDirStr.substr(0, 2) != "tf")
                continue;

            dirs.push_back(path + "/" + dirStr + "/" + innerDirStr);
            TSystemDirectory kineDir("KineDir", (path + "/" + dirStr + "/" + innerDirStr).data());
            auto kineFiles = kineDir.GetListOfFiles();
            for (auto kineFile : *kineFiles)
            {
                TString kinefileStr = ((TSystemFile *)kineFile)->GetName();
                if (kinefileStr.EndsWith("Kine.root") && kinefileStr.Contains("sgn"))
                {
                    kineNames.push_back(std::string(kinefileStr.Data()));
                }
            }
        }
    }

    // std::sort(dirs.begin(), dirs.end());

    int counter = 0;

    for (auto iDir{0}; iDir < dirs.size(); iDir++)
    {
        auto dir = dirs[iDir];
        auto kineName = kineNames[iDir];

        // if (counter > 50)
        //     continue;
        counter++;

        LOG(info) << "Processing: " << counter << ", dir: " << dir;
        LOG(info) << "Processing: " << counter << ", kine: " << kineName;
        std::string kine_file = dir + "/" + kineName;
        // Files
        auto fKine = TFile::Open(kine_file.data());
        std::vector<o2::MCTrack> *MCtracks = nullptr;

        if (!fKine)
            continue;
        auto treeMCTracks = (TTree *)fKine->Get("o2sim");
        // Setting branches
        treeMCTracks->SetBranchAddress("MCTrack", &MCtracks);
        // creating MC matrix
        auto nev = treeMCTracks->GetEntriesFast();
        nEvents += nev;
    }
    LOG(info) << "Number of events: " << nEvents;
}
#if !defined(CLING) || defined(ROOTCLING)
#include "CommonDataFormat/RangeReference.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#include "ITStracking/IOUtils.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"

#include <gsl/gsl>
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3D.h"
#include "TH1D.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TNtuple.h"
#include "CommonDataFormat/RangeReference.h"
#endif

using CompClusterExt = o2::itsmft::CompClusterExt;
// using

// original loop: checkClustersITS.C
void clusterTracksResiduals()
{
  TFile *outfile = TFile::Open("clusterResiduals.root", "recreate");
  TNtuple nt{"nt", "nt", "tid:ncl:clid:resy:resz:trx:try:trz:trphi:trpt:trp:trtgl"};
  std::vector<TH2F *> histsResZ(7);
  std::vector<TH2F *> histsResY(7);
  std::vector<TH2F *> histsResZvsZ(7);
  for (int iHist{0}; iHist < 7; ++iHist)
  {
    histsResZ[iHist] = new TH2F(Form("histResZ_Layer_%d", iHist), Form("histResZ_Layer_%d;#varphi;#Delta Z (cm)", iHist), 300, 0, TMath::TwoPi(), 300, -1, 1);
    histsResY[iHist] = new TH2F(Form("histResY_Layer_%d", iHist), Form("histResY_Layer_%d;#varphi;#Delta Y (cm)", iHist), 300, 0, TMath::TwoPi(), 300, -1, 1);
    histsResZvsZ[iHist] = new TH2F(Form("histResZvstan#lambda_Layer_%d", iHist), Form("histResZvstan#lambda_Layer_%d;tan#lambda (cm);#Delta Z (cm)", iHist), 300, -4, 4, 300, -1, 1);
  }
  o2::base::GeometryManager::loadGeometry("o2");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  // Topology dictionary
  o2::itsmft::TopologyDictionary mdict;
  mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));
  std::vector<int> runNumbers = {505669};
  for (auto &runNum : runNumbers)
  {
    std::ostringstream strDir;
    strDir << runNum;
    auto dir = strDir.str();

    std::string o2trac_its_file = dir + "/" + "o2trac_its.root";
    std::string o2clus_its_file = dir + "/" + "o2clus_its.root";

    // Files
    auto fITS = TFile::Open(o2trac_its_file.data());
    auto fITSclus = TFile::Open(o2clus_its_file.data());

    auto treeITS = (TTree *)fITS->Get("o2sim");
    auto treeITSclus = (TTree *)fITSclus->Get("o2sim");

    std::vector<o2::its::TrackITS> *ITStracks = nullptr;
    std::vector<int> *ITSTrackClusIdx = nullptr;

    // Clusters
    std::vector<CompClusterExt> *ITSclus = nullptr;
    std::vector<unsigned char> *ITSpatt = nullptr;

    // Setting branches
    treeITS->SetBranchAddress("ITSTrack", &ITStracks);
    treeITS->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);
    treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
    treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

    bool useITSonly = true;

    for (int frame = 0; frame < treeITS->GetEntriesFast(); frame++)
    {
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
          o2::dataformats::DCA *dca = new o2::dataformats::DCA();
          auto &clus = (*ITSclus)[(*ITSTrackClusIdx)[firstClus + icl]];
          auto layer = gman->getLayer(clus.getSensorID());
          TrackClus.push_back(clus);
          auto pattID = clus.getPatternID();
          o2::math_utils::Point3D<float> locXYZ;

          if (pattID != o2::itsmft::CompCluster::InvalidPatternID)
          {
            if (!mdict.isGroup(pattID))
            {
              locXYZ = mdict.getClusterCoordinates(clus);
            }
            else
            {
              o2::itsmft::ClusterPattern patt(pattIt);
              locXYZ = mdict.getClusterCoordinates(clus, patt);
            }
          }
          else
          {
            o2::itsmft::ClusterPattern patt(pattIt);
            locXYZ = mdict.getClusterCoordinates(clus, patt, false);
          }
          auto sensorID = clus.getSensorID();
          // Inverse transformation to the local --> tracking
          auto trkXYZ = gman->getMatrixT2L(sensorID) ^ locXYZ;
          // Transformation to the local --> global
          auto gloXYZ = gman->getMatrixL2G(sensorID) * locXYZ;
          auto alpha = gman->getSensorRefAlpha(sensorID);
          o2::dataformats::VertexBase vtx;
          vtx.setX(gloXYZ.x());
          vtx.setY(gloXYZ.y());
          vtx.setZ(gloXYZ.z());
          ITStrack.rotate(alpha);
          ITStrack.propagateToDCA(vtx, 2.f, dca);

          nt.Fill(iTrack, ncl, icl, dca->getY(), dca->getZ(), ITStrack.getX(), ITStrack.getY(), ITStrack.getZ(), ITStrack.getPhi(), ITStrack.getPt(), ITStrack.getP(), ITStrack.getTgl());
          delete dca;
        }
      } // track
    }   // frame
    treeITS->ResetBranchAddresses();
    treeITSclus->ResetBranchAddresses();
    fITS->Close();
    fITSclus->Close();
  }
  outfile->cd();
  for (int iLayer{0}; iLayer < 7; ++iLayer)
  {
    nt.Draw(Form("resz:trphi>>histResZ_Layer_%d", iLayer), Form("clid==%d&&TMath::Abs(resz)<1", iLayer), "goff");
    nt.Draw(Form("resy:trphi>>histResY_Layer_%d", iLayer), Form("clid==%d&&TMath::Abs(resz)<1", iLayer), "goff");
    nt.Draw(Form("resz:trtgl>>histResZvstan#lambda_Layer_%d", iLayer), Form("clid==%d&&TMath::Abs(resz)<1", iLayer), "goff");
  }

  outfile->Write();
  outfile->Close();
}
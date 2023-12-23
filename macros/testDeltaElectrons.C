#if !defined(CLING) || defined(ROOTCLING)
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "TSystemDirectory.h"
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#endif

// TMCProcess {
//   kPPrimary = 0, kPMultipleScattering = 1, kPCoulombScattering = 45, kPEnergyLoss = 2,
//   kPMagneticFieldL = 3, kPDecay = 4, kPPair = 5, kPCompton = 6,
//   kPPhotoelectric = 7, kPBrem = 8, kPDeltaRay = 9, kPAnnihilation = 10,
//   kPAnnihilationRest = 11, kPAnnihilationFlight = 12, kPHadronic = 13, kPEvaporation = 14,
//   kPNuclearFission = 15, kPNuclearAbsorption = 16, kPPbarAnnihilation = 17, kPNbarAnnihilation = 18,
//   kPNCapture = 19, kPHElastic = 20, kPHIElastic = 21, kPHCElastic = 22,
//   kPHInhelastic = 23, kPPhotonInhelastic = 24, kPMuonNuclear = 25, kPElectronNuclear = 26,
//   kPPositronNuclear = 27, kPPhotoNuclear = 46, kPTOFlimit = 28, kPPhotoFission = 29,
//   kPRayleigh = 30, kPNull = 31, kPStop = 32, kPLightAbsorption = 33,
//   kPLightDetection = 34, kPLightScattering = 35, kPLightWLShifting = 48, kStepMax = 36,
//   kPCerenkov = 37, kPFeedBackPhoton = 38, kPLightReflection = 39, kPLightRefraction = 40,
//   kPSynchrotron = 41, kPScintillation = 42, kPTransitionRadiation = 49, kPTransportation = 43,
//   kPUserDefined = 47, kPNoProcess = 44
// }

double calcRad(const o2::MCTrack &motherTrack)
{

    auto decLength = motherTrack.GetStartVertexCoordinatesX() * motherTrack.GetStartVertexCoordinatesX() +
                     motherTrack.GetStartVertexCoordinatesY() * motherTrack.GetStartVertexCoordinatesY();
    return sqrt(decLength);
}

void testDeltaElectrons()
{
    std::vector<TH1D *> hists(2);
    hists[0] = new TH1D("Delta rays w/ process disabled all", "; #delta-ray  Radius (cm) ; Counts", 200, 1, 50);
    hists[1] = new TH1D("Delta rays w/ process disabled ITS", "; delta-ray Radius (cm) ; Counts", 200, 1, 50);
    hists[2] = new TH1D("Delta rays w/ process enabled", "; delta-ray Radius (cm) ; Counts", 200, 1, 50);

    TH1D * hEnergy = new TH1D("Delta Energy distr", "; E (MeV) ; Counts", 500, 1e-3, 3e-1);
    TH2D *hXYCoord = new TH2D("XY coord", "; X (cm) ; Y (cm)", 600, -40, 40, 600, -40, 40);

    std::array<TString, 3> fileNames = {"/data/fmazzasc/its_data/sim/MB/tf1/sgn_1_Kine.root"};
    for (int j = 0; j <1; j++)
    {

        TFile file = TFile(fileNames[j]);
        TTree *tree = (TTree *)file.Get("o2sim");
        std::vector<o2::MCTrack> *MCtracks = nullptr;
        tree->SetBranchAddress("MCTrack", &MCtracks);
        auto nev = tree->GetEntriesFast();

        int counter = 0;
        for (int n = 0; n < nev; n++)
        { // loop over MC events
            tree->GetEvent(n);

            for (unsigned int i = 0; i < MCtracks->size(); i++)
            { // loop over MC tracks
                auto &track = (*MCtracks)[i];

                if (track.GetPdgCode() == 11 && track.leftTrace(0) && !track.leftTrace(1) && track.getProcess() == 9)
                {

                    counter++;
                    double rad = calcRad(track);
                    hEnergy->Fill(track.GetEnergy()*(1e3) - 0.511);
                    LOG(info) << "Radius: " << rad << " Energy: " << track.GetEnergy()*(1e3) - 0.511 << " Process: " << track.getProcess();
                    hXYCoord->Fill(track.GetStartVertexCoordinatesX(), track.GetStartVertexCoordinatesY());
                }
            }
        }
    }

    auto outFile = TFile::Open("DeltaElectrons_cdiff.root", "RECREATE");
    hEnergy->Write();
    hXYCoord->Write();
    outFile->Close();
}
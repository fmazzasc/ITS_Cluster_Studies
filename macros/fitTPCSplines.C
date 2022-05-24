#if !defined(CLING) || defined(ROOTCLING)
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "TSystemDirectory.h"
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TH2D.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "DataFormatsTPC/BetheBlochAleph.h"

#include <sstream>
#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using MCTrack = o2::MCTrack;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using namespace o2::tpc;

using Vec3 = ROOT::Math::SVector<double, 3>;

std::string getTFnumb(std::string &str)
{
    std::string final = "";
    std::string::iterator it;
    // Traverse the string
    for (it = str.begin(); it != str.end();
         it++)
    {
        auto c = *it;
        if (atoi(&c))
            final += c;
    }
    return final;
}

TF1 *fit(TH2D *histo);

double BetheBloch(double *x, double *p)
{
    return BetheBlochAleph(x[0], p[0], p[1], p[2], p[3], p[4]);
}

void fitTPCSplines()
{
    TH2D *pionTPC = new TH2D("pionTPC", "; #beta#gamma; dEdx", 100, 0.5, 8, 100, 0, 1000);
    TH2D *kaonTPC = new TH2D("kaonTPC", "; #beta#gamma; dEdx", 100, 0.4, 5, 100, 0, 1000);
    TH2D *protonTPC = new TH2D("protonTPC", "; #beta#gamma; dEdx", 100, 0.3, 5, 100, 0, 1000);
    TH2D *electronTPC = new TH2D("electronTPC", "; #beta#gamma; dEdx", 100, 1, 100, 100, 0, 1000);
    TH2D *he3TPC = new TH2D("he3TPC", "; #beta#gamma; dEdx", 100, 0.3, 4, 100, 0, 1000);
    TH2D *commonTPC = new TH2D("commonTPC", "; #beta#gamma; dEdx", 600, 0.25, 8, 100, 0, 1000);
    TH1D *momResolution = new TH1D("momResolution", "", 100, -1, 1);

    // std::string path = "/data/fmazzasc/its_data/MC/301004";
    std::vector<std::string> paths = {"/data/fmazzasc/its_data/sim/pi", "/data/fmazzasc/its_data/sim/k", "/data/fmazzasc/its_data/sim/p", "/data/fmazzasc/its_data/sim/he"};
    std::vector<std::string> dirs;

    for (auto &path : paths)
    {
        LOG(info) << path;
        TSystemDirectory dir("MyDir", path.data());
        auto files = dir.GetListOfFiles();
        for (auto fileObj : *files)
        {
            std::string file = ((TSystemFile *)fileObj)->GetName();
            if (file.substr(0, 6) == "o2_ctf" or file.substr(0, 2) == "tf")
                dirs.push_back(path + "/" + file);
            LOG(info) << path + "/" + file << " added";
        }
    }

    int counter = 0;

    for (auto &dir : dirs)
    {

        LOG(info) << "Processing dir: " << dir;
        std::string tf = getTFnumb(dir);
        std::string o2trac_tpc_file = dir + "/" + "tpctracks.root";
        std::string kine_file = dir + "/" + "sgn_Kine.root";

        TFile *fTPC = TFile::Open(o2trac_tpc_file.data());
        TFile *fMCTracks = TFile::Open(kine_file.data());

        if (!fMCTracks){
            kine_file = dir + "/" + "sgn_" + tf + "_Kine.root";
            fMCTracks = TFile::Open(kine_file.data());

        }

        if (!fTPC || !fMCTracks)
        {
            LOG(error) << "Failed to open file: " << o2trac_tpc_file << " or " << kine_file;
            continue;
        }

        auto treeMCTracks = (TTree *)fMCTracks->Get("o2sim");
        // TFile *fTPC = TFile::Open("tf1/tpctracks.root");
        auto treeTPC = (TTree *)fTPC->Get("tpcrec");

        // Labels
        std::vector<o2::MCTrack> *MCtracks = nullptr;
        std::vector<o2::MCCompLabel> *labTPCvec = nullptr;
        std::vector<o2::tpc::TrackTPC> *TPCtracks = nullptr;

        treeMCTracks->SetBranchAddress("MCTrack", &MCtracks);
        treeTPC->SetBranchAddress("TPCTracksMCTruth", &labTPCvec);
        treeTPC->SetBranchAddress("TPCTracks", &TPCtracks);

        // fill MC matrix
        std::vector<std::vector<o2::MCTrack>> mcTracksMatrix;
        auto nev = treeMCTracks->GetEntriesFast();

        mcTracksMatrix.resize(nev);
        for (int n = 0; n < nev; n++)
        { // loop over MC events
            treeMCTracks->GetEvent(n);

            mcTracksMatrix[n].resize(MCtracks->size());
            for (unsigned int mcI{0}; mcI < MCtracks->size(); ++mcI)
            {
                mcTracksMatrix[n][mcI] = MCtracks->at(mcI);
            }
        }

        for (int frame = 0; frame < treeTPC->GetEntriesFast(); frame++)
        {
            if (!treeTPC->GetEvent(frame))
            {
                continue;
            }
            for (unsigned int iTrack{0}; iTrack < labTPCvec->size(); ++iTrack)
            {

                auto lab = labTPCvec->at(iTrack);
                auto track = TPCtracks->at(iTrack);

                int trackID, evID, srcID;
                bool fake;
                lab.get(trackID, evID, srcID, fake);
                if (!lab.isNoise() && lab.isValid() && lab.isCorrect() && !lab.isFake())
                {
                    auto trackPDG = mcTracksMatrix[evID][trackID].GetPdgCode();

                    if (trackPDG == 1000020030)
                        track.setAbsCharge(2);
                    
                    momResolution->Fill((track.getP() - mcTracksMatrix[evID][trackID].GetP()) / mcTracksMatrix[evID][trackID].GetP());
                    

                    if ((abs(track.getP() - mcTracksMatrix[evID][trackID].GetP()))/mcTracksMatrix[evID][trackID].GetP() > 0.2)
                        continue;

                    if (abs(trackPDG) == 211)
                    {

                        double mass = TDatabasePDG::Instance()->GetParticle(trackPDG)->Mass();
                        pionTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                        commonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                    }

                    if (abs(trackPDG) == 321)
                    {
                        double mass = TDatabasePDG::Instance()->GetParticle(trackPDG)->Mass();
                        kaonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                        commonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                    }

                    if (abs(trackPDG) == 2212)
                    {

                        momResolution->Fill((track.getP() - mcTracksMatrix[evID][trackID].GetP()) / mcTracksMatrix[evID][trackID].GetP());
                        double mass = TDatabasePDG::Instance()->GetParticle(trackPDG)->Mass();
                        protonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                        commonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                    }
                    if (abs(trackPDG) == 11)
                    {
                        momResolution->Fill((track.getP() - mcTracksMatrix[evID][trackID].GetP()) / mcTracksMatrix[evID][trackID].GetP());
                        double mass = TDatabasePDG::Instance()->GetParticle(trackPDG)->Mass();
                        electronTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                        commonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                    }
                    if (abs(trackPDG) == 1000020030)
                    {
                        momResolution->Fill((track.getP() - mcTracksMatrix[evID][trackID].GetP()) / mcTracksMatrix[evID][trackID].GetP());
                        double mass = 2.80839160743;
                        he3TPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                        commonTPC->Fill(track.getP() / mass, track.getdEdx().dEdxTotTPC);
                    }
                }
            }
        }
    }

    auto outFile = TFile::Open("TPCSplines.root", "RECREATE");
    momResolution->Write();

    TF1 *bethePi = fit(pionTPC);
    TF1 *betheP = fit(protonTPC);
    TF1 *betheE = fit(electronTPC);
    TF1 *betheHe3 = fit(he3TPC);
    TF1 *betheK = fit(kaonTPC);

    TF1 *betheCommon = fit(commonTPC);

    bethePi->SetLineColor(kMagenta);
    betheP->SetLineColor(bethePi->GetLineColor() + 5);
    betheE->SetLineColor(bethePi->GetLineColor() + 10);
    betheHe3->SetLineColor(bethePi->GetLineColor() + 15);
    betheK->SetLineColor(bethePi->GetLineColor() + 20);

    betheCommon->SetLineColor(kBlack);

    pionTPC->Write();
    bethePi->Write();
    auto cv = new TCanvas("cv_pi", "cv_pi", 800, 600);
    cv->cd();
    pionTPC->Draw("colz");
    bethePi->Draw("same");
    betheP->Draw("same");
    auto legend0 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend0->AddEntry(pionTPC, "#pi");
    legend0->AddEntry(bethePi, "BetheBlochAleph #pi");
    legend0->AddEntry(betheP, "BetheBlochAleph p");
    legend0->Draw();
    cv->Write();

    auto cv_pr = new TCanvas("cv_pr", "cv_pr", 800, 600);
    cv_pr->cd();
    protonTPC->Draw("colz");
    betheP->Draw("same");
    bethePi->Draw("same");
    // betheCommon->Draw("same");
    auto legend1 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend1->AddEntry(protonTPC, "p");
    legend1->AddEntry(bethePi, "BetheBlochAleph #pi");
    legend1->AddEntry(betheP, "BetheBlochAleph p");
    legend1->Draw();

    cv_pr->Write();

    auto cv_e = new TCanvas("cv_e", "cv_e", 800, 600);
    cv_e->cd();
    electronTPC->Draw("colz");
    bethePi->Draw("same");
    betheE->Draw("same");
    betheP->Draw("same");

    auto legend2 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend2->AddEntry(electronTPC, "p");
    legend2->AddEntry(betheE, "BetheBlochAleph e");
    legend2->AddEntry(bethePi, "BetheBlochAleph #pi");
    legend2->AddEntry(betheP, "BetheBlochAleph p");
    legend2->Draw();
    cv_e->Write();

    auto cv_he3 = new TCanvas("cv_he3", "cv_he3", 800, 600);
    cv_he3->cd();
    he3TPC->Draw("colz");
    betheHe3->Draw("same");
    auto legend3 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend3->AddEntry(he3TPC, "He3");
    legend3->AddEntry(betheHe3, "BetheBlochAleph He3");
    legend3->Draw();
    cv_he3->Write();

    auto cv_k = new TCanvas("cv_k", "cv_k", 800, 600);
    cv_k->cd();
    kaonTPC->Draw("colz");
    betheK->Draw("same");
    auto legend4 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend4->AddEntry(kaonTPC, "K");
    legend4->AddEntry(betheK, "BetheBlochAleph K");
    legend4->Draw();
    cv_k->Write();

    auto cv_bethe = new TCanvas("cv_bethe", "cv_bethe", 800, 600);
    cv_bethe->cd();
    bethePi->Draw();
    betheP->Draw("same");
    betheE->Draw("same");
    betheHe3->Draw("same");
    betheK->Draw("same");
    auto legend5 = new TLegend(0.1, 0.7, 0.48, 0.9);
    legend5->AddEntry(betheE, "BetheBlochAleph e");
    legend5->AddEntry(betheP, "BetheBlochAleph p");
    legend5->AddEntry(bethePi, "BetheBlochAleph #pi");
    legend5->AddEntry(betheHe3, "BetheBlochAleph He3");
    legend5->AddEntry(betheK, "BetheBlochAleph K");
    legend5->Draw();
    cv_bethe->Write();

    outFile->Close();

    LOG(info) << " ----------------------------";
    LOG(info) << "BetheBlochAleph for electrons: " << betheE->GetParameter(0) << ", " << betheE->GetParameter(1) << ", " << betheE->GetParameter(2) << ", " << betheE->GetParameter(3) << ", " << betheE->GetParameter(4);
    LOG(info) << "BetheBlochAleph for pions: " << bethePi->GetParameter(0) << ", " << bethePi->GetParameter(1) << ", " << bethePi->GetParameter(2) << ", " << bethePi->GetParameter(3) << ", " << bethePi->GetParameter(4);
    LOG(info) << "BetheBlochAleph for protons: " << betheP->GetParameter(0) << ", " << betheP->GetParameter(1) << ", " << betheP->GetParameter(2) << ", " << betheP->GetParameter(3) << ", " << betheP->GetParameter(4);
    LOG(info) << "BetheBlochAleph for kaons: " << betheK->GetParameter(0) << ", " << betheK->GetParameter(1) << ", " << betheK->GetParameter(2) << ", " << betheK->GetParameter(3) << ", " << betheK->GetParameter(4);
    LOG(info) << "BetheBlochAleph for he3: " << betheHe3->GetParameter(0) << ", " << betheHe3->GetParameter(1) << ", " << betheHe3->GetParameter(2) << ", " << betheHe3->GetParameter(3) << ", " << betheHe3->GetParameter(4);
}

TF1 *fit(TH2D *histo)
{
    histo->FitSlicesY(0, 0, -1, 0);
    TH1D *mean = (TH1D *)gDirectory->Get(Form("%s_1", histo->GetName()));
    TH1D *sigma = (TH1D *)gDirectory->Get(Form("%s_2", histo->GetName()));
    sigma->Divide(mean);
    sigma->Fit("pol0", "Q");
    TF1 *bethe = new TF1(Form("%s_bethe", histo->GetName()), BetheBloch, 0.3, 100, 5);
    bethe->SetNpx(10000);
    double starting_pars[5]{-166.11733, -0.11020473, 0.10851357, 2.7018593, -0.087597824};
    bethe->SetParameters(starting_pars);
    bethe->SetParLimits(0, -150, 0);
    bethe->SetParLimits(1, -2, 0);
    bethe->SetParLimits(2, 0, 20);
    bethe->SetParLimits(3, 0, 10);
    bethe->SetParLimits(4, -2, 2);


    mean->Fit(bethe, "", "", histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
    mean->Write(Form("%s_mean", histo->GetName()));
    mean->SetStats(0);
    sigma->Write(Form("%s_sigma", histo->GetName()));
    return bethe;
}
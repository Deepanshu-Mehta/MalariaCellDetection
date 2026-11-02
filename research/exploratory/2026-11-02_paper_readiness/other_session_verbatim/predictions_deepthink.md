# Exploratory, unregistered. Written 2026-11-02 BEFORE computing any of the below.
P1 Hand-made dark-spot feature (no training): NIH AUC 0.90-0.97; MP-IDB AUC at or above DANet's 0.88.
P2 E7 control: some Uninfected-labelled, flagged-as-parasitized cells sit in healthy-only patients (at least 10%).
P3 MP-IDB: oracle threshold lifts mean BACC by 5+ points; 5-checkpoint ensemble AUC beats mean single AUC by 0.01+.
P4 Date-clustered bootstrap widens E9 and E1 CIs by 1.2x or more versus patient-clustered.
P5 Healthy-only cell false-positive rate implies minimum detectable parasitemia of about 1% at threshold 0.5.

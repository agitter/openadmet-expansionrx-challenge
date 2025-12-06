<!-- Converted from: Expansion-CDD-webinar.pdf -->
<!-- Original file type: .pdf -->

ExpansionRx OpenADMET Blind Challenge

October 27, 2025

© 2025 OpenADMET

1

AI in Drug Discovery Has Generated Tremendous Interest

© 2025 OpenADMET

2

Data is Critical to Machine Learning

© 2025 OpenADMET

3

Current Public Datasets Are Highly Flawed

Data collected from dozens of papers, each of which uses different experimental conditions

Public data contains numerous curation errors

Dynamic range is often unrealistic

Endpoints are not relevant

© 2025 OpenADMET

4

Data Generation and Community Engagement Improve ML in Drug Discovery

Improving the understanding of factors that create the most significant delays in drug discovery projects

● Absorption
● Metabolism
● Excretion
● Toxicity

Generating large high-quality datasets

● Assay data
● X-ray crystallographic and cryo-EM structures of protein-ligand complexes

Acting as a catalyst for the community

● Conducting blind challenges
● Implementing best practices as open-source software (ANVIL)
● Using community feedback to drive data generation and promote best practices

© 2025 OpenADMET

5

https://openadmet.org

© 2025 OpenADMET

6

ExpansionRx OpenADMET Blind Challenge

ADMET data from ExpansionRx drug discovery programs
● Realistic time time splits of training and test data

9 distinct assay endpoints
● LogD
● Kinetic solubility (KSOL)
● Human liver microsome (HLM) stability
● Mouse liver microsome (MLM) stability
● Caco-2 Papp A>B
● Caco-2 Efflux Ratio
● Mouse Plasma Protein Binding (MPPB)
● Mouse Brain Protein Binding (MBPB)
● Mouse Gastrocnemius Muscle Binding (MGMB)

© 2025 OpenADMET

7

Expansion RX & The challenge concept

Compound Design, Triage, and Prio.

Compound Synthesis

1st Pass ADME
LogD, Solubility, Caco2 perm.

RNA Binding
SPR

RNA + Protein
SPR

2nd Pass ADME
H/R/MLM Stab., Protein binding

Functional
cell assays

Cytotoxicity

Further ADME / Selectivity
CYP, CEREP, CardioTox, etc

In vivo PK
Mouse/Rat

In vivo PK/PD

© 2025 OpenADMET

8

8

Putting ADMET prediction to the test

The OpenADMET + ExpansionRx Blind Challenge

- A community-driven effort to benchmark predictive model for drug discovery
- Participants are tasked with developing and submitting their models on a Hugging Face platform

Create a Hugging face account

Download the public training dataset

- Hugging Face platform
- CDD platform

Train your model

📅 Timeline:

○ October 27: Start of the challenge
○ Oct-Nov: Online Q&A sessions + support via Discord
○ January 19, 2026: Submission closes & final

leaderboard

○ January 26, 2026: Winners announced

Submit your results & check the leaderboard!

© 2025 OpenADMET

9

Putting ADMET prediction to the test

More details on the evaluation process

○ The leaderboard is live and will be updated after every submission.
○ You can submit as many times as you want throughout the challenge (up to a reasonable limit),

with submissions tracked based on Hugging Face username.

○ Live leaderboard ≠ final leaderboard: Evaluation done on different halves of the test set.
○ Endpoint ranking is based on MAE, while global ranking based on a macro-averaged relative

absolute error (MA-RAE) across the endpoints.

○ Anonymous participation is possible by providing an Alias.

🏆 Prizes:
➔ All participants are eligible to co-author a future publication detailing the

results of the challenge (contact info provided with submission).

➔ Participants with notable performance will also be invited to present their

work at a special workshop after the conclusion of the challenge.

© 2025 OpenADMET

10

Ready to join the challenge?

Let’s do a tour through the Hugging Face platform 🤩
https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

Step-by-step tutorial also available on Github: https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial/tree/main

© 2025 OpenADMET

11


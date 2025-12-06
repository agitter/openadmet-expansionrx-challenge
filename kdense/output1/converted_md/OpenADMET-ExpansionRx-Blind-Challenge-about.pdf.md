<!-- Converted from: OpenADMET-ExpansionRx-Blind-Challenge-about.pdf -->
<!-- Original file type: .pdf -->

12/5/25, 2:08 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

Spaces

openadmet / OpenADMET-ExpansionRx-Challenge

like

33

Running

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

1/5

12/5/25, 2:08 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

Welcome to the OpenADMET + ExpansionRx Blind
Challenge!

Your task is to develop and submit predictive models for key ADMET properties on a
blinded test set of real world drug discovery data 🧑‍🔬

Go to the Leaderboard to check out how the challenge is going. To participate, head

out to the Submit tab and upload your results as a  CSV  file.

We released an intermediate leaderboard on December 2nd, with submissions

evaluated against the full blinded test set. Check it out on the OpenADMET blog!

📖 About 🚀 Leaderboard ✉ Submit 🛠 FAQ

💊 OpenADMET + ExpansionRx

Computational Blind Challenge in ADMET

This challenge is a community-driven initiative to benchmark predictive models for

ADMET properties in drug discovery, hosted by OpenADMET in collaboration with

ExpansionRx.

Why are ADMET properties important in drug
discovery?

Small molecules continue to be the bricks and mortar of drug discovery globally,

accounting for ~75% of FDA approvals over the last decade. Oral bioavailability,

easily tunable properties, modulation of a wide range of mechanisms, and ease of

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

2/5

12/5/25, 2:08 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

manufacturing make small molecules highly attractive as therapeutic agents.

Moreover, emerging small molecule modalities such as degraders, expression

modulators, molecular glues, and antibody-drug conjugates (to name a few) have

vastly expanded what we thought small molecules were capable of. It is fairly

difficult to predict the lifetime and distribution of small molecules within the body.

Additionally, interaction with off-targets can cause safety issues and toxicity.

Collectively these Absorption, Distribution, Metabolism, Excretion, Toxicology--or

ADMET--properties sit in the middle of the assay cascade and can make or break

preclinical candidate molecules. OpenADMET aims to address these challenges

through an open science effort to build predictive models of ADMET properties by

characterizing the proteins and mechanisms that give rise to these properties

through integrated structural biology, high throughput experimentation and

integrative computational models. Read more about our strategy to transform drug

discovery on our website. Critical to our mission is developing open datasets and

running community blind challenges to assess the current state of the art in ADMET

modeling. Building on the sucess of the recent ASAP-Polaris-OpenADMET blind

challenge in computational methods for drug discovery, we bring you a brand new

challenge in collaboration with ExpansionRx. During a recent series of drug

discovery campaigns for RNA mediated diseases, ExpansionRX collected a variety

of ADMET data for off-targets and properties of interest, which they are generously

sharing with the community for this challenge.

🧪 The Challenge

Participants will be tasked with solving real-world ADMET prediction problems

ExpansionRx faced during lead optimization. Specifically, you will be asked to

predict the ADMET properties of late-stage molecules based on earlier-stage data

from the same campaigns. For this challenge we selected nine (9) crucial endpoints

for the community to predict:

LogD

Kinetic Solubility KSOL: uM

Mouse Liver Microsomal (MLM) CLint: mL/min/kg

Human Liver Microsomal (HLM) Clint: mL/min/kg

Caco-2 Efflux Ratio

Caco-2 Papp A>B (10^-6 cm/s)

Mouse Plasma Protein Binding (MPPB): % Unbound

Mouse Brain Protein Binding (MBPB): % Unbound

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

3/5

12/5/25, 2:08 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

Mouse Gastrocnemius Muscle Binding (MGMB): % Unbound

Find more information about these endpoints on our blog.

UPDATE: The Challenge is now live! Data available at the following Hugging Face

Datasets

Training: https://huggingface.co/datasets/openadmet/openadmet-

expansionrx-challenge-train-data

Test: https://huggingface.co/datasets/openadmet/openadmet-expansionrx-

challenge-test-data-blinded

You can also watch a Webinar where we introduce the challenge, hosted by

Collaborative Drug Discovery (CDD).

UPDATE: OpenEye Cadence Molecular Sciences is generously providing access to

their Toolkit for interested participants during the duration of the challenge.

Request access by filling out this Google Form.

✅ How to Participate
1. Register: Create an account with Hugging Face.

2. Walk through the tutorials: We have prepared a Tutorial showing how to train a
model and submit to the leaderboard.

3. Download the Public Dataset: Download the ExpansionRx training and blinded
test sets from Hugging Face.

4. Train Your Model: Use the provided training data for each ADMET property of
your choice.

5. Submit Predictions: Follow the instructions in the Submit tab to upload your
predictions.

6. Join the discussion on the Challenge Discord!

📊 Data:

The training set contains the following parameters:

Column

Unit

Molecule
Name

Smiles

Typ
e

str

str

Description

Identifier for the
molecule

Text representation of
the 2D molecular
structure

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

4/5

12/5/25, 2:08 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

LogD

KSol

float

LogD

uM

float

Kinetic Solubility

MLM CLint

mL/min/kg

float Mouse Liver Microsomal

HLM CLint

mL/min/kg

float

Human Liver
Microsomal

Caco-2
Permeability
Efflux

Caco-2
Permeability
Papp A>B

float

Caco-2 Permeability
Efflux Ratio

10^-6 cm/s

float

Caco-2 Permeability
Papp A>B

MPPB

% Unbound

float

MBPB

% Unbound

float

MGMB

% Unbound

float

Mouse Plasma Protein
Binding

Mouse Brain Protein
Binding

Mouse Gastrocnemius
Muscle Binding

You can download the training data from the Hugging Face dataset.

The test set will remained blinded until the challenge submission deadline. You will

be tasked with predicting the same set of ADMET endpoints for the test set

molecules.

The training and blinded test set will also be made available on the CDD Vault. An

account to access the CDD Vault can be requested by filling out this form. Note that

by joining the Vault your account will be visible to other participants so this

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

5/5


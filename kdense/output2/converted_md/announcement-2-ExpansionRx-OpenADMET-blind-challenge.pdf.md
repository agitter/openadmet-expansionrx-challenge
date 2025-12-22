<!-- Converted from: announcement-2-ExpansionRx-OpenADMET-blind-challenge.pdf -->
<!-- Original file type: .pdf -->

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

Announcement 2: ExpansionRx -
OpenADMET Blind Challenge

OpenADMET
14 Oct 2025 — 6 min read

Additional details on the coming challenge

Absorption, Distribution, Metabolism, Excretion and Toxicity–ADMET–properties can

make or break preclinical and clinical development of small molecules. At

OpenADMET we address the unpredictable nature of these properties through open

science, generating experimental data and building predictive models of ADMET

properties. A key component of these efforts is running community blind challenges

to benchmark the current state of the art in ADMET modeling.

On September 16, we announced the ExpansionRx-OpenADMET blind challenge in

partnership with Expansion Therapeutics and HuggingFace. Participants will predict

multiple ADMET properties for a diverse set of molecules collected from a real-world

drug discovery campaign targeting RNA-mediated diseases, including Myotonic

Dystrophy, ALS, and Dementia. Below, we provide more details on the coming

challenge, how to participate, and the dataset and evaluation criteria we will use to

select the winners.

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

1/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

The Challenge

Participants will be tasked with solving real-world ADMET prediction problems

ExpansionRx faced during lead optimization. Specifically, you will be asked to predict

the ADMET properties of late-stage molecules based on earlier-stage data from the

same campaigns.

This type of time-split challenge mimics the real-world task of leveraging historical

optimization data to improve decision-making, a core component of multi-parameter

optimization in drug development and a key point at which models can add value. Not

an easy task!

You will be provided with molecules from early in the optimization campaign to train

your model (training split is released on HuggingFace) and provided with molecules

from later in the lead-opt campaign to then submit your predictions on (test split).

You can find a small sample of the training data here:

https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-

teaser

Check out the HuggingFace Space where the challenge will be hosted:

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

Come and discuss the challenge on the OpenADMET Discord.

OpenADMET will also be submitting a series of baselines to support participants,

which will be shared using our open-source infrastructure. Stay tuned for updates on

the Discord Channel!

Endpoints included in this challenge

You will apply your models for the prediction of nine (9) crucial ADMET endpoints:

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

2/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

LogD: Measures compound’s lipophilicity at a specific pH. Drugs typically  fall into

a well defined LogD range that balances aqueous solubility with membrane

permeability, making understanding changes in LogD across a chemical series

important to medicinal chemists. Additionally, assessing LogD of candidate

molecules can suggest whether candidate molecules are "efficient" for their

lipophilicity (lipophilicity generally has a linear relationship with affinity).

Kinetic Solubility (KSOL): Measures how much a compound can be dissolved

under non-equilibrium conditions. Helps screen compounds that will fail due to

poor absorption or low bioavailability.

Human Liver Microsomal (HLM) stability: Supports understanding of a

compound's susceptibility to liver metabolism and can be used to predict in-vivo

clearance of a candidate molecule. Measured using human liver microsomes and
reported as intrinsic clearance for the compound Clint (mL/min/kg).

Mouse Liver Microsomal (MLM) stability: Also used to predict in-vivo clearance

of a candidate. The study of both MLM and RLM can provide a more

comprehensive understanding of a compound's metabolic profile and how a

compound might behave in multi-species preclinical development.

Caco-2 Papp A>B:  Measures the rate of flux of a compound across polarized

Caco-2 cell monolayers from the apical (intestinal lumen-facing side) to

basolateral (blood-facing side). This effectively mimics the absorption of a drug

across the intestinal wall.

Caco-2 Efflux Ratio: Measures the rate of flux of a compound across polarized

Caco-2 cell monolayers. The Efflux Ratio is determined by a ratio of the apparent

permeability coefficient (Papp) in both directions. Ratios of ~1 indicate that a

compound primarily traverses the cell membrane via passive (diffusional)

transport. Ratios > 2 generally indicate active transport of the compound across

the cellular membrane by membrane bound transporters (e.g. efflux by p-

glycoprotein).

Mouse Plasma Protein Binding (MPPB): Determines the concentration of free

drug in plasma (as % Unbound). Drugs that are not bound to plasma can bind to

target proteins and yield the desired therapeutic effect, making this parameter

crucial to understanding drug distribution.

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

3/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

Mouse Brain Protein Binding (MBPB): This measures the fraction of a drug not

bound to proteins within brain tissue. The unbound fraction of a drug in the brain

is considered pharmacologically active and able to interact with central nervous

system (CNS) targets. MBPB helps assess CNS drug exposure and potential

efficacy or side effects for neuroactive compounds. Reported as % Unbound.

Mouse Gastrocnemius Muscle Binding (MGMB):. This reflects the amount of

drug free to act within skeletal muscle tissue, which is important for drugs

targeting peripheral or muscular conditions (very important for the DM1

indication). Reported as % Unbound

Judging Criteria and Rules

We welcome submissions of any kind, including machine learning and physics-based

approaches. You can employ pre-training approaches as you see fit. You are also free

to reuse data from one portion of the challenge for others if it will assist you. You are

free to incorporate data from external sources (e.g, public repositories) into your

models and submissions.

You can submit to the HuggingFace portal multiple times (within a once-a-day rate

limit, but up to the judges' final discretion), however, only your latest submission will

count towards the intermediate and final leaderboards. In the spirit of our mission to

promote open science, we encourage participants to share the code used to create

their submissions. If not possible due to IP or other constraints, we require at a

minimum that participants provide a short written report on the methodology used for

submissions (based on the template here).

We encourage teams in industry to compete. We understand this isn’t always

possible in public-facing challenges, therefore, we will allow teams to compete

anonymously as a group using a pseudonym or alias. A HuggingFace user account is

required to compete and track multiple submissions. However, if you choose the

anonymous participation option, the associated user account will not be visible to

other participants. If choosing to compete anonymously, you can choose to reveal

yourself post challenge if you wish.

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

4/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

The endpoints will be judged individually by mean absolute error (MAE), while the

overall leaderboard will be judged by the macro-averaged relative absolute error (MA-

RAE). For endpoints that are not already on a log scale (e.g, LogD), they will be

transformed to a log scale to minimize the impact of outliers on evaluation.  Relative

absolute error (RAE) normalizes the MAE to the dynamic range of the test data,

making the RAE comparable between endpoints, unlike MAE. We will estimate errors

on the metrics using bootstrapping and use the statistical testing workflow outlined in

this paper to determine if model performance is statistically distinct.

During the challenge, we will run a live leaderboard that will be updated after every

submission. This live leaderboard will reflect the result of evaluating your submissions

against a partial fraction of the test set (validation split). A final leaderboard will be

released after the submission window is closed, showing the final submissions

evaluated against the full test set.  The competition is run with an open science ethos,

and all queries and issues are encouraged to be discussed openly. The OMSF code of

conduct applies to all communication around the challenge. The organizers reserve

the right to exclude participants deemed to be acting in bad faith with respect to the

integrity of the competition.

We plan to release a summary paper detailing the challenge results, to which all

eligible teams will be invited to participate.  Hope to see you there!

Timeline

The challenge timeline is as follows:

September 12: First challenge announcement

October 10: Second announcement and sample data release

October 27: Challenge starts

October-November: Online Q&A sessions and support via the Discord channel

January 19, 2026: Submission closes and final leaderboard released

January 26, 2026: Winners announced

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

5/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

Post-Challenge Virtual workshop

Participants with notable performance or learning outcomes will be invited to present

their work at a special blind-challenge workshop, hosted by OpenADMET, to share

their findings.

How to get started

Check out the teaser data and get excited! We will release a full tutorial prior to the

start of the competition.

Questions or Ideas?

We’d love to hear from you! Whether you want to learn more, have ideas for future

challenges, or wish to contribute data.

Join the OpenADMET Discord or contact us at openadmet@omsf.io.

Let’s work together to transform ADMET modeling and accelerate drug discovery!

Acknowledgements

We gratefully acknowledge Jon Ainsley, Andrew Good, Elyse Bourque,

Lakshminarayana Vogeti, Renato Skerlj, Tiansheng Wang, and Mark Ledeboer for

generously providing the Expansion Therapeutics dataset used in this challenge as an

in-kind contribution to OpenADMET!

READ MORE

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

6/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

OpenADMET -
ExpansionRx
Blind
Challenge

Concerning
Uncertainty

Or, how do you
know what you…

Intermediate
leaderboard relea…

By OpenADMET — 11
Nov 2025

By OpenADMET — 02
Dec 2025

Time For a
New
Adventure

Author: Pat Walters,
Chief Scientist,…

By OpenADMET — 15
Sep 2025

Announcement
1:
ExpansionRx-
OpenADMET
Blind
Challenge

Small molecules
continue to be the…

By OpenADMET — 16
Sep 2025

Sign up

Powered by Ghost

OpenADMET

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

7/8

12/5/25, 2:29 PM

Announcement 2: ExpansionRx - OpenADMET Blind Challenge

jamie@example.com

Subscribe

https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/

8/8


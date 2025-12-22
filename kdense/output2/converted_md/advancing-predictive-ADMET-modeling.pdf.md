<!-- Converted from: advancing-predictive-ADMET-modeling.pdf -->
<!-- Original file type: .pdf -->

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

Search models, datasets, users...

Back to Articles

Advancing Predictive ADMET Modeling
Through Community-Driven Science: The
ExpansionRx-OpenADMET Blind
Challenge

Community Article

Published October 27, 2025

Upvote 12

+6

Georgia Channing
Follow

cgeorgiaw

Hugo MacDermott

hmacdope-omsf

Follow

hugging-science

hugging-science

We’re excited to announce that the ExpansionRx-OpenADMET Blind Challenge is

now live on Hugging Face!

This community challenge aims to advance benchmarking of predictive models for

Absorption, Distribution, Metabolism, Excretion, and Toxicology (ADMET)

properties by providing a high-quality experimental dataset that participants can

use to train and evaluate their models.

The Challenge

Small molecules continue to be the foundation of modern drug discovery,

representing nearly 75% of FDA drug approvals in the last decade. Their ease of

synthesis and tunable properties make them highly versatile therapeutic agents.

However, predicting how these molecules behave in the body (including how long

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

1/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

they persist, where they go, and whether they interact with other targets), is

extremely challenging. Collectively, these ADMET properties sit at the heart of the

assay cascade and can make or break preclinical candidate molecules.

OpenADMET is an open-science initiative that aims to tackle these challenges by

integrating structural biology, high-throughput experimentation, and

computational modeling to improve ADMET prediction (read more about their

strategy here). A key part of these eﬀorts is organizing blind challenges to

benchmark the current state of predictive modeling on real, high-quality datasets.

In partnership with ExpansionRx, this challenge invites participants to solve

realistic ADMET prediction problems that ExpansionRx encountered during lead

optimization, by predicting the properties of late-stage molecules based on earlier-

stage data from the same campaigns, across nine endpoints.

A high-quality dataset of ADMET endpoints

Expansion Therapeutics recently prosecuted several drug discovery programs

targeting RNA-mediated diseases, including Myotonic Dystrophy (DM1),

Amyotrophic Lateral Sclerosis (ALS), and Dementia. During optimization, they

generated a wealth of high-quality ADMET data oﬀ-targets and properties of

interest. Now, they’ve made the bold and generous decision to open-source their

high-quality ADMET dataset for public use and beneﬁt.

The dataset includes over 7,000 small molecules measured across multiple ADMET

assays, and was divided into:

A training set, which will be available to all participants during the challenge

A blinded test set, which will be used to score ﬁnal predictions after

submission.

Participants will train models on the training data, submit predictions for the

molecules in the blinded test set, and have their models evaluated on the unseen

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

2/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

datapoints. ADMET Endpoints to predict There are a total of nine ADMET

endpoints that participant is tasked with predicting:

LogD: Measures compound’s lipophilicity at a speciﬁc pH, which helps

understand the balance of aqueous solubility with membrane permeability.

Kinetic Solubility KSOL: Quantiﬁes how much a compound can be dissolved

under non-equilibrium conditions (in µM units).

Human Liver Microsomal (HLM) CLint: Helps predict in vivo liver

metabolism and clearance (in mL/min/kg).

Mouse Liver Microsomal (MLM) stability: Analogous to HLM, studying MLM

can provide a more comprehensive understanding of a compound's metabolic

proﬁle and how a compound might behave in multi-species preclinical

development. (in mL/min/kg).

Caco-2 Papp A>B: Measures the rate of ﬂux of a compound across polarized

Caco-2 cell monolayers from the apical (intestinal lumen-facing side) to

basolateral (blood-facing side), eﬀectively mimicking the absorption of a drug

across the intestinal wall (measured in 10^-6 cm/s).

Caco-2 Eﬄux Ratio: Models intestinal absorption by measuring ﬂux across

polarized Caco-2 cell monolayers.

Mouse Plasma Protein Binding (MPPB): Determines the concentration of free

drug in plasma (as % Unbound).

Mouse Brain Protein Binding (MBPB): Measures the fraction of drug not

bound to proteins within brain tissue (as % Unbound):

Mouse Gastrocnemius Muscle Binding (MGMB): Reﬂects the amount of drug

free to act within skeletal muscle tissue, which is important for drugs targeting

peripheral or muscular conditions (as % Unbound)

How to get started

The challenge is run from a Hugging Face Space. Explore the challenge space here.

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

3/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

Feel free to ask any questions on our community Discord. The organizers and other

participants will be able to assist you.

Run through the tutorial:

We have a set of tutorials available for you to get acquainted with the challenge

here.

We recommend that all participants run through the tutorial to fully understand the

endpoints and submission process.

Access the training set:

Click here to see the dataset on the Hugging Face Hub, or you can easily download

the clean version of the training dataset using the Hugging Face Datasets library:

from datasets import load_dataset

ds = load_dataset("openadmet/openadmet-expansionrx-challenge-train-data

df = ds["train"].to_pandas()





And to download the raw version of the training dataset that contains out of bounds

measurements.

from datasets import load_dataset

ds = load_dataset("openadmet/openadmet-expansionrx-challenge-train-data

df = ds["train"].to_pandas()





Access the blinded test set:

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

4/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

See the dataset on the Hub here.

The test set contains only the SMILES strings for you to perform inference on before

submitting to the challenge space.

Similarly to the training set you can access the blinded test data using the dataset

library:

from datasets import load_dataset

ds = load_dataset("openadmet/openadmet-expansionrx-challenge-test-data

df = ds["test"].to_pandas()





By taking part in this blind challenge, you’ll contribute to open, reproducible

ADMET modeling and help benchmark the next generation of predictive models for

drug discovery!

Questions or Ideas?

OpenADMET would love to hear from you whether you’re:

Interested in participating

Have ideas for future challenges

Want to contribute data

Join the OpenADMET Discord or email openadmet@omsf.io.

Community

Edit

Preview

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

5/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

Start discussing this article

Tap or paste here to upload images

Comment

System theme

Company

TOS

Privacy

About

Jobs

Website

Models

Datasets

Spaces

Pricing

Docs

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

6/7

12/5/25, 2:27 PM

Advancing Predictive ADMET Modeling Through Community-Driven Science: The ExpansionRx-OpenADMET Blind Challenge

https://huggingface.co/blog/hugging-science/the-expansionrx-openadmet-blind-challenge

7/7


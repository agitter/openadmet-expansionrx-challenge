# Assessing AI Scientists on the OpenADMET + ExpansionRx Blind Challenge

## Introduction

## Methods

## Results

The full results of my 11 submissions are below.
After each submission, I copied the output from the [leaderboard](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) before making the next submission.
The ranks are not comparable because the number of other strong submissions on the leaderboard changed over time.
The context column is a summary of the context provided in the prompt and dataset, which is detailed in the models' respective subdirectories in this repository and described in the Methods.
Two submissions are duplicates or near duplicates.
Submissions 3 and 4 used the same prompt.
Submission 11 is a resubmission of submission 3 to create my final challenge entry.
Overall, Kosmos with context about previous tutorials and runs was my best performing model.

| Submission | Model | Prompt | rank | MA-RAE | R2 | Spearman R | Kendall's Tau | submission time | context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | K-Dense | prompt1 | 80 | 0.73 +/- 0.03 | 0.30 +/- 0.05 | 0.66 +/- 0.03 | 0.49 +/- 0.02 | 2025-12-06 17:42:24+00:00 | Tutorials, leaderboard, AW Kosmos run |
| 2 | Biomni | prompt1 | 136 | 0.94 +/- 0.03 | -0.07 +/- 0.06 | 0.46 +/- 0.04 | 0.33 +/- 0.03 | 2025-12-07 03:18:52+00:00 | Tutorials, leaderboard, AW Kosmos run |
| 3 | Kosmos | prompt2 | 42 | 0.68 +/- 0.03 | 0.46 +/- 0.04 | 0.70 +/- 0.02 | 0.53 +/- 0.02 | 2025-12-08 03:39:59+00:00 | Tutorials, leaderboard, AW Kosmos run |
| 4 | Kosmos | prompt1 | 82 | 0.74 +/- 0.03 | 0.34 +/- 0.05 | 0.65 +/- 0.03 | 0.47 +/- 0.02 | 2025-12-08 18:11:15+00:00 | Tutorials, leaderboard, AW Kosmos run |
| 5 | Kosmos | prompt3 | 70 | 0.73 +/- 0.03 | 0.36 +/- 0.04 | 0.66 +/- 0.03 | 0.49 +/- 0.02 | 2025-12-09 03:40:34+00:00 | Leaderboard, previous outputs from this repo |
| 6 | Heureka | prompt1 | 156 | 0.78 +/- 0.03 | 0.29 +/- 0.04 | 0.62 +/- 0.03 | 0.45 +/- 0.02 | 2025-12-20 15:45:44+00:00 | Leaderboard |
| 7 | Heureka | prompt2 | 196 | 0.93 +/- 0.03 | -0.03 +/- 0.05 | 0.47 +/- 0.04 | 0.33 +/- 0.03 | 2025-12-21 23:53:27+00:00 | Dataset and descriptions |
| 8 | K-Dense | prompt2 | 147 | 0.76 +/- 0.03 | 0.33 +/- 0.04 | 0.62 +/- 0.03 | 0.45 +/- 0.02 | 2025-12-22 14:02:12+00:00 | Dataset and descriptions |
| 9 | Biomni | prompt2 | 211 | 1.05 +/- 0.04 | -0.32 +/- 0.09 | 0.35 +/- 0.04 | 0.24 +/- 0.03 | 2025-12-24 18:32:58+00:00 | Dataset and descriptions |
| 10 | Kosmos | prompt4 | 198 | 0.89 +/- 0.04 | 0.06 +/- 0.06 | 0.59 +/- 0.03 | 0.42 +/- 0.02 | 2025-12-27 14:04:23+00:00 | Dataset and descriptions |
| 11 | Kosmos | prompt2 | 91 | 0.68 +/- 0.03 | 0.46 +/- 0.04 | 0.70 +/- 0.02 | 0.53 +/- 0.02 | 2026-01-06 15:24:22+00:00 | Tutorials, leaderboard, AW Kosmos run |

The figure below examines how the different models perform when they are given minimal context, only the ADMET datasets and descriptions about the competition.
That is, they are not provided with any previous exploratory analyses or reports that guide them towards particular predictive modeling strategies.
In this setting, K-Dense has an advantage, narrowly per some metrics and more substantially per R<sup>2</sup>.
![Performance with dataset only as context](results/dataset_only_relative_performance.png)

Looking across all models and contexts, we can examine how adding additional context helps the AI scientists.
In all cases it provides a benefit, often a considerable benefit.
I did not perform detailed ablations so I cannot tell what specifically was most valuable for the predictive modeling.
Even with the additional context, Biomni lags behind the other three models.
![Performance by model with varying context](results/context_comparison_per_model.png)

I ran Kosmos in a unique setting where I gave it a zip of the current repository as context, which included results from the initial K-Dense, Biomni, and Kosmos runs.
That context was roughly equivalent to the other informative context I provided previously.

![Kosmos performance with varying context](results/kosmos_context_effect.png)

## User experience
After running four different AI scientists, I've accumulated some subjective thoughts about each.

Say (at least) one nice thing about:
- **Biomni**: It is fairly open and transparent, which I value. The planning and analysis is mostly linear, even when backtracking and updating plans, which makes it possible to attempt to scroll through all the generated code and output to understand what analysis was done.
- **Heureka**: I like interface of creating a project, writing a prompt, and then having a batch job run in the background once there was heavy work to complete. The output files were well-organized.
- **K-Dense**: The performance without context guiding the modeling strategy is worth mentioning. The [scientific skills](https://github.com/K-Dense-AI/claude-scientific-skills) are available outside the web app. The output files were well-organized. I especially liked the hierarchical layout and ability to easily download everything in batch.
- **Kosmos**: It blasts analyses at your problem. The magnitude of computation dwarfs the other tools I have tried, and this can lead to secondary or tertiary explorations that are unexplored by the other tools.

Constructive feedback for:
- **Biomni**: Users cannot copy and paste long input prompts. Long inputs are treated as attachments, so the user has to break up the input into chunks of a few sentences and paste them in bit-by-bit. Users also cannot download the logs and file outputs without refreshing the page and reloading the past session.
- **Heureka**: The [terms](https://heurekalabsco.github.io/terms/terms.html) concern me with respect to the language about granting the Company a license to publicly display User Content. The input prompt length limit is too short. Attempting to zip and download dozens of files in a single batch failed.
- **K-Dense**: The interface hangs when submitting a prompt with many attached files.
- **Kosmos**: The overall structure of a Kosmos analysis is rigid: four discoveries with a text-based report as the main deliverable. That can be restrictive for a project like this where the goal is to find the best single modeling strategy and produce the best possible predictions for an input set of compounds. It is difficult to find and download the best output `.csv` file across the multiple analyses in the report. The report references analysis tasks, but those tasks may not produce an output `.csv` file. I cannot find a way to download all the data artifacts and code generated. There are so many parallel analyses attempted, some of which are dead ends, that it feels hopeless to trace through them to understand in full what modeling was done.

## Discussion

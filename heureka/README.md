# Heureka Labs
[Heureka Labs](https://www.heurekalabs.co/) has a few products: an AI Research Companion (ARC), QuartzReport, and CarbonDraft.
QuartzReport looked most relevant for dataset analysis, but the website had [broken links](https://www.quartzreport.com/pricing) so I couldn't immediately figure out whether it was live.
Instead, I signed up for ARC.
I'm a nerd, so I read the [terms](https://heurekalabsco.github.io/terms/terms.html) first.
One thing that caught my attention, besides the license being "solely for your own personal, noncommercial use" is regarding user content:
> You hereby grant (and you represent and warrant that you have the right to grant) to Company an irrevocable, nonexclusive, royalty-free and fully paid, worldwide license to reproduce, distribute, **publicly display** and perform, prepare derivative works of, incorporate into other works, and otherwise use and exploit your User Content, and to grant sublicenses of the foregoing rights, solely for the purposes of including your User Content in the Site. You hereby irrevocably waive (and agree to cause to be waived) any claims and assertions of moral rights or attribution with respect to your User Content.

Be careful about what data you upload!

I created an ARC project to upload files in my dataests and then started a conversation with my prompt with Extended Reasoning on and Memory off.
Note that `prompt1.txt` does not describe all of the attached files. This is because this prompt consumbed 4990/5000 characters, and I could not extend it.
ARC prepared an analysis plan and queried "Should I deploy the analysis agent for this?", to which I responded "Yes".

ARC submitted a batch job that ran in the background.
When it finished, my project had a directory with 49 files.
Attempting to zip and download all 49 files failed with "Error downloading files: Failed to fetch", so I had to download them in small batches, which was cumbersome.
It included the input datasets, which I did not download.
The results are in the directory [`output1`](output1).
The file `train_features.pkl` exceeded GitHub's file size limit of 100 MB so I zipped it to `train_features.zip`.
I appreciated the job summary report that cleared stated `admet_predictions_final.csv` is the final submission and the model weights files.

`prompt2.txt` was updated to remove context and datasets related to previous ADMET modeling and exploratory analyses that might biases the modeling decisions.
I once again created an ARC project to upload files and then started a conversation with my prompt with Extended Reasoning on and Memory off.
ARC queried "Should I deploy the analysis agent for this comprehensive ADMET modeling task?", and I responded "Yes".

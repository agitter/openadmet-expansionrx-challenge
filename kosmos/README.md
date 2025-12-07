# Kosmos
With my Kosmos credits expiring, it was time to start on the OpenADMET + ExpansionRx Blind Challenge.

![Kosmost plan expiration](kosmos-plan-expiration.png)

There was no time to iterate because a single Kosmos run may take many hours.
Therefore, I decided to learn from the best and build on [Andrew White's existing Kosmos run](https://dev.platform.edisonscientific.com/kosmos/8208890b-d46b-402d-b17f-6d69063f9cb1).
His original prompt is provided as `andrew-white-prompt.txt`.
I modified it into my prompt.
The main strategy was to provide additional resources, including Andrew's own Kosmos report, as input datasets.

The first run failed 80% of the way through after running for about 7 hours.
I'm not sure what happened, the last task was in the "success" state but no more tasks were initialized.

![Kosmost failure](kosmos-failed.png)

I resubmitted another run with the same dataset and prompt, which succeeded.
The primary results are in the directory [`output2`](output2) because there is no easy way to download all of the artifacts.
The full set of artifacts are available at [Edison Scientific](https://platform.edisonscientific.com/kosmos/368185cd-80ca-4689-9b0d-1f5a1868c20c).
In general, I have found that Kosmos emphasizes the final human-readable report but makes it difficult to find final output files.
It was unclear what csv file to use for the competition predictions.
In the list of artifacts, the last csv file created was `submission_hybrid_7task_optimized.csv` from task 53, so I downloaded that file and used it for the submission.
However, it wasn't referenced in the final report.

(**TODO:** make the Edison Scientific link public)

<!-- Converted from: OpenADMET-ExpansionRx-Blind-Challenge-FAQ.pdf -->
<!-- Original file type: .pdf -->

12/5/25, 2:09 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

Spaces

openadmet / OpenADMET-ExpansionRx-Challenge

like

33

Running

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

1/5

12/5/25, 2:09 PM

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

1. How long does it normally take for submissions to reach the leaderboard?

At most, this should take 2 to 3 minutes. If it's taking longer, please ping us on

Discord and let us know.

2. The leaderboard isn't updating.

Early in the challenge we had a problem with the leaderboard not updating. We

believe this has been fixed. Please reach out on Discord if you have an issue.

2. My submission didn't upload, what's wrong.

Please check your submission and confirm that the column names are the same

as those in the test set file. If you run into an issue, please reach out on Discord,

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

'd b h

t h l

2/5

12/5/25, 2:09 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

we'd be happy to help.

3. I only want to submit for a one endpoint, what should I do.

Right now, you have to submit all columns. If you'd like to just submit LogD, or

some other column, please include the other columns and put zeros for all

values. This should still rank you on the LogD leaderboard. Your rankings on the

other leaderboards will be low.

4. What is the formula for macro-averaged relative absolute error (MA-RAE)? What
is this relative to?

The code we use to compute all the metrics is available here

5. Organizers, where are you?

We try to answer questions on Discord as quickly as possible, but we
occassionally need to sleep. 🙂

6. Do I have to upload my model to HuggingFace?

No, you only need to upload the results.

7. Do I have to make the code for my model public?

No, while we love open source, you don't have to make your model public. We

would appreciate a brief description of how you built your models. NOTE: You

must provide a link to a report or github repository before the challenge deadline

in order to be considered for the final leaderboard.

8. Can I use data beyond the training set to train my model.

Yes, absolutely, you're free to use any data you'd like to train your model.

9. How are you handling log transforms of zero values?

Please see the function function clip_and_log_transform. This function adds 1 to

both the training and test data before doing the log transform.

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

3/5

12/5/25, 2:09 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

4/5

12/5/25, 2:09 PM

OpenADMET ExpansionRx Blind Challenge - a Hugging Face Space by openadmet

Use via API

· Built with Gradio

· Settings

https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

5/5


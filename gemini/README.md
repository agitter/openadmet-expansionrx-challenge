# Gemini
AI Scientists should perform better than the general LLMs they use in the background, right?
Right???
Let's run the experiment!

I booted up Gemini 3 Pro from <https://gemini.google.com/> on December 19, 2025.
I could only upload 10 files, so I had to exclude 4 files that the other tools received.
`prompt1.txt` shows the excluded files and the text prompt, which was otherwise similar to what the other tools were given for instructions.

Gemini returned Python code for me to run in my environment (`output1.py`).
I responded "Your instructions were to provide predictions in a csv (same format as training) for the test compounds."
It worked for a while, but then I got a "Something went wrong message" so I tried again with the same response.
It happened again on the second and third attempts.
Then, I modified my response to "Your instructions were to provide predictions in a csv (same format as training) for the test compounds. Make sure to only use dependencies you have available."
The initial code output showed it hit a roadblock:
```
Available libraries: {'rdkit': True, 'lightgbm': False, 'xgboost': False, 'sklearn': True}
```
This still did not produce output.
Last attempt:
> Your instructions were to provide predictions in a csv (same format as training) for the test compounds. I absolutely need a csv file and cannot tolerate failure to provide one. Use in-context learning if you cannot write and execute the code successfully. Do. Not. FAIL!

Still no output.
The initial response is in `output1.pdf`.

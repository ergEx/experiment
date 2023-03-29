# Experiment
Experiment and analysis for the first behavioural experiment of ergEx

_This experiment should be run using PsychoPy_

This is a general list of what happens when the experiment is run without including the mathematical context (will be completed after all algorithms are ran through)

"Press Run"
Fill in:

![image](https://user-images.githubusercontent.com/122382899/227543168-c2256df1-ae09-4c91-b3cf-b063b3c22769.png)
- Subject ID
- Select whether this experiment is to be run in test-mode or not
- Select whether calibration has to be run
- Indicate whether the passive phase of the experiment is to be run
- Indicate whether the active phase of the experiment is to be run
- Indicate whether the session should be started
- Indicate whether questionnaires should be completed
- Indicate whether instructions should be given
"Press OK"

Now instructions on the passive experiment will be given, the participant can click through them by themselves (if indicated that instructions should be given)

After the passive experiment will be run. Consists of three times 45 learning images and 15 query choices. After each run, the wealth is reset to 1000. 
  (Learning images are shown to the participant, after which the wealth changes, this teaches the participant the worth of the image, 
  query choices are implemented to assess whether the participant is learning throughout the different trials)

When the passive experiment is finished, the active phase is initiated, instructions on the active experiment are given, the participant can click through them by 
themselves (if indicated that instructions should be given)

Now the active experiment is run. It consists of 160 trials where four fractals are shown. The participant is to choose between the two fractals on either the left or 
right (gamble pairs), now a coin toss will decide whether the upper of lower fractal will be the one influencing the participants wealth. The choices that the 
participant makes here will impact their wealth, their wealth will not be reset anymore. 

  Important to note: to make the payscheme executable, an upper and lower bound are applied for the wealth. When either bound is neared, the fractals shown are adjusted 
  accordingly to prevent the participant from crossing the bound, crossing the bound is impossible as the participant will not get an option that would make their wealth 
  cross the bound.

After the completion of the active experiment of session 1, there is a mandatory break for the participant. After the break time has passed, the second session will be
initiated. The second session is build up of the exact same trial order as the first session.

After the completion of the second session, the participant will be requested to answer a total of four questionnaires (if indicated that questionnaires should be 
completed). The questionnaires are used to assess the general riskaversion of the participant.

After the questionnaires are completed, the files with results are created according to the BIDS structure. The files created are both input files and output files
for both sessions:
- Input files:
  - The sequence of fractals shown during the passive phase (type: TSV file)
  - The combinations of fractals shown during the active phase which give the option for a good gamble (type: TSV file)
  - The combinations of fractals shown during the active phase which give the option for a neutral gamble (type: TSV file)
  - The combinations of fractals shown during the active phase which give the option for a bad gamble (type: TSV file)
 - Output files: 
    A file with all events recorded during the completion of the task, separated per session (type: TSV file)

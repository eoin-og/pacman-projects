A few notes explaining the implementation.

*** In many of the function I changed the 'state' and 'action' arguments to 's' and 'a' so that they would more closely reflect the functions from lecture notes.

===============================
Question 1
===============================
This was completed by doing the following:

Adding a calculate_values function which gets the max q value of any non-terminal state.

Finishing computeQValueFromValues. The return statement might be a little difficult to read but I tried to make it as close as possible to the actual formula from the lecture notes.

Finishing computeActionFromValues so it gets the maximum value for any possible action from a given state.



===============================
Question 2
===============================
The noise was set to 0. This removed the possibility of randomly landing on one of the negative terminals. This meant the agent could now cross the bridge since the discounted score was worth it when the risk was removed.


===============================
Question 3
===============================
Broadly this problem was solved by:

Keeping the noise low when you wanted the agent to take the quicker path (no risk of landing on a negative terminal)
Keeping the discount ~1 when you want the agent to go to the higher value terminal (since this means future rewards are more highly valued.)
Setting the discount ~0 and the living reward high to stop the agent terminating (since it values living more than terminating)


===============================
Question 4
===============================
This problem was solved by:

Finishing computeValueFromQValues so it returns the max qvalue for all the legal actions for a given state. An exception is added for a ValueError which will occur if there are no legal actions (i.e. you are in a terminal state).

computeActionFromQValues. This is more or less the same as in Question 1 but uses random.choice to break tie breaks between equally good actions.

update implements the update function from the lecture notes.


===============================
Question 5
===============================
This problem was solved by updating the getAction function to sometimes randomly select a legal action - instead of always choosing the best. Whether or not the action was selected randomly was decided using util.coinflip.


===============================
Question 6
===============================
I don't think there was a possible value that would allow the agent to reach the other side of the bridge. You need to raise the value of the agent to get it try out new states which is a necessity for crossing the bridge. However this also means the agent becomes very likely to hit a negative terminal (which lowers it's chances of getting across the bridge and makes it learn negative values for the bridge.) 


===============================
Question 7
===============================

===============================
Question 8
===============================
This problem was solved by:

Implementing getQValue. This gets a state value by summing each feature by it's corresponding weight.

Implementing update so that it updated each weight according to the functions from the lecture notes.



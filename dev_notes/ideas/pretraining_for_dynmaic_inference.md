comment preparer l'inference dynamique durant le pretraining?



l'inference dynamique signifie le model decide de lookup : input (ram), medium term memory (ram) or long term memory  (disk), maybe more in the future. il y a dautre lenment lie au dynamisme telque le nombre de step, le nombre de latent la memoire de travail court terme (registre par assimilation au cpu).



chaque memoire a temps d'access different et un mecanisme de stoakage different.



mecanisme de stockage:

mecanisme de stockage differentd uembedding space en generale plus basse

- input very low dimension to medium eg. alphabetic level token simple input as possible we delegate information enrichement to the model

- long term memory medium dimension (not sure how to handle indexing retrieval, wonder if FAISS,HNSW,PQ,Annoy,LSH could fit this use case)

- medium memory full dimenssion same as model ndim 



retrievial mechanism:

- latent conpute direct access as register, and cross attend on other

- memory (memdium/short) and input via cross attention, and projection to ndim compute dimenssion

since at inference those memory has different speed/latency we can mimic during training by delaying the retreiver information few step later in the compute



retrievial delay simulation :

- memory : 95% direct, 5% delay of 1 step, we could increase delay probability with the oldest in memory simulate pagination delay

- longterm : 5% direct, 55% 1 step, 40% 2-3steps or more, the simulate multicache level (if too complex to simulated delay base usage frequency could make it random)   

- input : 80% direct, 20% 1 step delay random, is to simulate pagination delay (we could also use access frequency but I'm not we will have enough pass as inference to compute frequency)



note: when I say memory without specifying the type I'm refring to medium memory 

note: does delay value are just guess by me to express the ideas we should not exctly follow them



query specilization:

at inference model shouldn't behave like all data are evailable

model should learn specialized query

- approche 1: prediction probe on attention query that predict either : mem, longterm, input - the class is determined by where most of the attention where given and loss penalise attention given out of that range. predictability of query target will help to decide how to handle the query during inference

- approche 2: projecting to 3 different type of queries, having al tree more memory trigger all the time is far from what we observe from everyday intelligence, we could add the probe tat predict if the queried data were useful or not (given few or a lot of attention)

note: I haven't mention query on the latent (self attention) since we looking to external nkowdlege

note: query on the latent could be treated as the same query on memory merging self and cross attention, this unify the 2 but spread compute capability, having self attention attention allow more focused information process/transformation compute separated than cross attention focus on retrieval compute 



at inference we will need :

- RL for better finetune and optimize the dynamic behavior

- we will a good way to estimation latent contribution in the final ouput eg. via gradient diff (not sure of how that express utility), via ablation (expensive), still looking for solutions.

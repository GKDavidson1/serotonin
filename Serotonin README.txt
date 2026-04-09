Serotonin

pgACC

awake or anaesthesia 

BBO open [VS] BBO Serotonin

Will it work if BBO is closed?

If BBO increases bloodflow, then perhaps a simple scalar can map closed to open??
CBF from BALLOON scaled by BBO/C scalar or liner function

POST-Process in "Workbench"



Serotonin: what happens under highest and lowest levels
	Do so for both excite and inhib


Background and STD_noise need to optimised!

1282ms for FMRI



consider setting ndma_da_grad to 1 : this removes the effect of dopamine

b1 changes the distribution of the weights in FLN


Bold Free Parameters
	kappa
	gamma
	tau
	alpha
	E0



e_grad_min: 0.15-0.25

I_background_e: 140-620

I_background_i: 120-300

mu_ee: >1.3

mu_ie:

At egradmin: 0.235

1.35 1.8 	+
1.4  2.0-2.2 	+
1.45 2.2 	++
1.45 2.3	+
1.5  2.4-2.6	+
1.55 2.6-2.7	++
1.55 2.8	+
1.6  2.8	+?
1.6  2.9 	+
1.65 3.0	+?
1.65 3.X	???



When optimising for 5HT case options:
	
	1) Try to fit to 5HT target data
	
	2) Create a difference vector based on the changes from NS->5HT
		-Fit to effects
		-Fit correlation 








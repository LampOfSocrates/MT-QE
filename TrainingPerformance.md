Heron - 10% data
	7.8K training data
	Train batch size = 8 , 2 workers
	Val and test batch size = 4 

	2.52 batch/s
	1 epoch	= 985 batch ( 7.8K/8)
	6min32s per epoch
	7.654 GB 
	needs to wipe gpu between epocsh. A suddent jump in GPU towards end of epoch

Lit - 100% data
	78K training data
	Train batch size = 16 , 2 workers
	Val and test batch size = 16, 2 workers

	1.56 batch/s
	1 epoch	= 4927 batch  ( 78K/16)
	51min per epoch

	13-14GB steady usage
	Fails in the 3071th batch 

	Setting to 4 workers seems to have made no diff. 
	
during eda we found that 90% of the data has less than 50 words , 99% of the data has than 100 wordsSo we normally work with smaller number of words in each sentence This translates into into less tokens for sentence and eventually much faster speed of training than the whole dataset sceanrio where we pad everything up to 233 tokens each 

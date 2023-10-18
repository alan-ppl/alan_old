def get_posteriors(T=0):
    posteriors = [{"N": 50,   "K": 30, "VAR_SIZE": "WIDE",   "LOC_VAR": 1.0,   "T_rws": T*2 , "T_vi": T//4,  "T_mcmc": T*4,  "T_lang": T*1},
              
              # best with T=2000
              {"N": 5000, "K": 3,  "VAR_SIZE": "WIDE",   "LOC_VAR": 1.0,   "T_rws": T*30, "T_vi": T*50,  "T_mcmc": T*50,  "T_lang": T*50},
              {"N": 5000, "K": 3,  "VAR_SIZE": "WIDE",   "LOC_VAR": 150.0, "T_rws": T*25, "T_vi": T*50,  "T_mcmc": T*50,  "T_lang": T*50},
              {"N": 5000, "K": 3,  "VAR_SIZE": "NARROW", "LOC_VAR": 1.0,   "T_rws": T*30, "T_vi": T*10,  "T_mcmc": T*50,  "T_lang": T*50},
              {"N": 5000, "K": 3,  "VAR_SIZE": "NARROW", "LOC_VAR": 150.0, "T_rws": T*20, "T_vi": T*20,  "T_mcmc": T*20,  "T_lang": T*20},
              
              # for vi and mcmc it DOES work with T*250 but trying T*500 with fingers crossed
              {"N": 5000, "K": 5,  "VAR_SIZE": "WIDE",   "LOC_VAR": 1.0,   "T_rws": T*25, "T_vi": T*500,  "T_mcmc": T*500,  "T_lang": T*10},
              {"N": 5000, "K": 5,  "VAR_SIZE": "WIDE",   "LOC_VAR": 150.0, "T_rws": T*20, "T_vi": T*500,  "T_mcmc": T*500,  "T_lang": T*10},
              {"N": 5000, "K": 5,  "VAR_SIZE": "NARROW", "LOC_VAR": 1.0,   "T_rws": T*20, "T_vi": T*500,  "T_mcmc": T*500,  "T_lang": T*10},
              {"N": 5000, "K": 5,  "VAR_SIZE": "NARROW", "LOC_VAR": 150.0, "T_rws": T*25, "T_vi": T*500,  "T_mcmc": T*500,  "T_lang": T*10},

              {"N": 5000, "K": 10, "VAR_SIZE": "WIDE",   "LOC_VAR": 1.0,   "T_rws": T*3,  "T_vi": T*3,   "T_mcmc": T*3,  "T_lang": T*3},
              {"N": 5000, "K": 10, "VAR_SIZE": "WIDE",   "LOC_VAR": 150.0, "T_rws": T*3,  "T_vi": T*3,   "T_mcmc": T*3,  "T_lang": T*3},
              {"N": 5000, "K": 10, "VAR_SIZE": "NARROW", "LOC_VAR": 1.0,   "T_rws": T*3,  "T_vi": T*3,   "T_mcmc": T*3,  "T_lang": T*3},
              {"N": 5000, "K": 10, "VAR_SIZE": "NARROW", "LOC_VAR": 150.0, "T_rws": T*12, "T_vi": T*30,  "T_mcmc": T*30, "T_lang": T*30},

              # best with T=2000*30
              {"N": 50,   "K": 5,  "VAR_SIZE": "WIDE",   "LOC_VAR": 1.0,   "T_rws": T , "T_vi": T//500,  "T_mcmc": T//50,  "T_lang": T*5},
              ]
    
    return posteriors
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 11:53:39 2020

@author: Ogawa
"""

import scraper as gs 
import pandas as pd 


path = "C:/Users/Ogawa/Desktop/Project_inpt_2020/chromedriver"

df = gs.get_jobs('Machine Learning Engineer',1000, False, path, 20)

df.to_csv('ML_jobs.csv', index = False)
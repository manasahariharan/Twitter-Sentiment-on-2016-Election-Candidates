import pandas as pd 
import GetOldTweets3 as got
from  data_collect_1 import gettweetdata, main_tweets, label_tweets
import os
#usecols = ['id','text'],
path = 'C:/Users/manas/OneDrive/Documents/578/project/projectdata/tagged_data/'
pro1 = pd.read_csv(path + 'pro_clinton_half1.csv', usecols = ['id','text','user_screen_name','hashtags'], 
	encoding = 'utf-8',lineterminator='\n')
print(len(pro1))
pro1 = pro1.append(pd.read_csv(path + 'pro_clinton_half2.csv',  usecols = ['id','text','user_screen_name','hashtags'], 
	encoding = 'utf-8',lineterminator='\n'))

print(len(pro1))
pro1 = pro1.drop_duplicates(keep='first')
pro1len = len(pro1)
print(pro1len)
pro1['pro_clinton'] = True

anti1 = pd.read_csv(path + 'anti_clinton_half1.csv', usecols = ['id','text','user_screen_name','hashtags'], 
	encoding = 'utf-8',lineterminator='\n')
print(len(anti1))
anti1 = anti1.append(pd.read_csv(path + 'anti_clinton_half2.csv', usecols = ['id','text','user_screen_name','hashtags'], 
	encoding = 'utf-8',lineterminator='\n'))
print(len(anti1))
anti1 = anti1.drop_duplicates(keep='first')
anti1['pro_clinton'] = False
print(len(anti1))
#anti1 = anti1.sample(pro1len)
#pro1 = pro1.sample(270000)
print(len(pro1))
pro1 = pro1.append(anti1)
pro1 = pro1.drop_duplicates(subset = ['id'], keep = False)
pro1 = pro1.sample(frac=1).reset_index(drop=True)
print(pro1.head(10))
print(len(pro1))
print(pro1['pro_clinton'].value_counts())
with open(path+"tagged_data.csv", mode='w', newline='\n', encoding = 'utf-8') as f:
    pro1.to_csv(f, line_terminator='\n', encoding='utf-8', index = False)



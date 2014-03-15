from __future__ import division
import pandas as pd
from lookup_kaggle_to_kenpom import team_lookup as kag_to_kp
from lookup_kenpom_to_kaggle import team_lookup as kp_to_kag
import re,sys
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk import edit_distance


class PredictNCAA():
    
    def __init__(self):
        self.year_lookup = {'N':'2009','O':'2010','P':'2011','Q':'2012','R':'2013'}
        self.teams_kaggle = pd.read_csv('./teams.csv')
        self.feature_names = ['Rank','Pyth','AdjO','AdjOR','AdjD','AdjDR',
                            'AdjT','AdjTR','Luck','LuckR','PythSOS','PythSOSR',
                            'OppO','OppOR','OppD','OppDR','PythNCSOS','PythNCSOSR']
   
    #Stores all data into dictionary____________________________________________
    def load_data(self):
        self.all_data = {}
        for year in range(2003,2015):
            kenpom_data = pd.read_csv('./%s_data.csv'%str(year),skiprows=1)
            kenpom_data.Team = kenpom_data.Team.apply(lambda x: re.sub(r'[0-9]*','',str(x)).strip().strip('.'))
            self.all_data[year]={}
            for id in self.teams_kaggle.id:
                name = kag_to_kp[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()]
                if isinstance(name,list): name = name[0]
                if len(kenpom_data.Team[kenpom_data.Team==name]) == 0 and isinstance(kag_to_kp[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()],list): 
                    name = kag_to_kp[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()][1]
                if len(kenpom_data.Team[kenpom_data.Team==name]) == 0: 
                    name = self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()
                if len(kenpom_data.Team[kenpom_data.Team==name]) != 0:
                    self.all_data[year][id]={}
                    self.all_data[year][id]['Name']=name
                    for column_name in kenpom_data.columns:
                        self.all_data[year][id][column_name]=kenpom_data[column_name][kenpom_data['Team']==name].item()
                else:
                    #print name+" not found in year "+str(year)
                    pass
      
    #Calculates standardized features over full data____________________________              
    def standardize_feats(self):
        self.means = {}
        self.stdevs = {}
        for feat in self.feature_names:
            all_vals = []
            for year in self.all_data.keys():
                for teamid in self.all_data[year].keys():
                    all_vals.append(float(self.all_data[year][teamid][feat]))
            self.means[feat] = np.mean(all_vals)
            self.stdevs[feat] = np.std(all_vals)
       
    #Used to find the teamname with smallest edit distance______________________ 
    def get_teamname(self,teamname):
        try:
            return kp_to_kag[teamname]
        except KeyError:
            name,val = '',100
            for team in self.teams_kaggle.name:
                newval = edit_distance(team,teamname)
                if newval < val:
                    name = team
                    val = newval
            return name
        
    #Trains a Logistic Regression model_________________________________________                
    def train_LR(self):
        X = []
        y = []
        for year in range(2003,2013):
            with open('./%s_tour.csv'%str(year),'r') as f:
                for line in f:
                    line = line.split(',')
                    team1 = line[0]
                    team2 = line[1]
                    team1wins = line[2]
                    if team1 not in self.teams_kaggle.name.values:
                        team1 = self.get_teamname(team1)
                    team1id = self.teams_kaggle.id[self.teams_kaggle.name==team1].values[0]
                    if team2 not in self.teams_kaggle.name.values:
                        team2 = self.get_teamname(team2)
                    team2id = self.teams_kaggle.id[self.teams_kaggle.name==team2].values[0]
                    team1feats = [(float(self.all_data[year][team1id][x]) - self.means[x])/self.stdevs[x] 
                                                    for x in self.feature_names]
                    team2feats = [(float(self.all_data[year][team2id][x]) - self.means[x])/self.stdevs[x]
                                                    for x in self.feature_names]
                    game_feats = team1feats + team2feats
                    X.append(game_feats)
                    y.append(team1wins)
        self.X = X
        self.y = y
                        
        self.LR = LogisticRegression()
        self.LR.fit(self.X,self.y)
        
    #Predicts the probabilistic outcome of a single game________________________
    def predict_game(self,team1name,team2name):
        if team1name not in self.teams_kaggle.name.values:
            team1name = self.get_teamname(team1name)
        team1id = self.teams_kaggle.id[self.teams_kaggle.name==team1name].values[0]
        if team2name not in self.teams_kaggle.name.values:
            team2name = self.get_teamname(team2name)
        team2id = self.teams_kaggle.id[self.teams_kaggle.name==team2name].values[0]
        year=2014
        team1feats = [(float(self.all_data[year][team1id][x]) - self.means[x])/self.stdevs[x] 
                            for x in self.feature_names]
        team2feats = [(float(self.all_data[year][team2id][x]) - self.means[x])/self.stdevs[x]
                            for x in self.feature_names]
        x = team1feats + team2feats
        pred = self.LR.predict_proba(x)
        return pred

if __name__ == '__main__':
    team1 = sys.argv[1]
    team2 = sys.argv[2]
    a = PredictNCAA()
    a.load_data()
    a.standardize_feats()
    a.train_LR()
    pred = a.predict_game(team1,team2)[0]
    print "Percent chances of winning:"
    print ("{team1}: %0.2f"%(pred[1]*100)).format(team1=team1)+"%"
    print ("{team2}: %0.2f"%(pred[0]*100)).format(team2=team2)+"%"


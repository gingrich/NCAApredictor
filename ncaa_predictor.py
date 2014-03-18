from __future__ import division
import pandas as pd
from lookup_kaggle_to_kenpom import team_lookup as kag_to_kp
from lookup_kenpom_to_kaggle import team_lookup as kp_to_kag
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    def training_testing_split(self):
        X = []
        y = []
        for year in range(2003,2013):
            with open('./%s_tour.csv'%str(year),'r') as f:
                num_zeros = 0
                num_ones = 0
                for line in f:
                    line = line.split(',')
                    team1 = line[0]
                    team2 = line[1]
                    team1wins = int(line[2].strip())
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
                    #game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
                    if num_ones > num_zeros:
                        if team1wins:
                            game_feats = team2feats + team1feats
                            actual_label = 0
                        else:
                            game_feats = team1feats + team2feats
                            actual_label = 0
                    else: 
                        game_feats = team1feats + team2feats
                        actual_label = team1wins
                    if actual_label: num_ones += 1
                    else: num_zeros += 1
                            
                    X.append(game_feats)
                    y.append(actual_label)
        N = int(.7*len(X))
        self.X_train = X[:N]
        self.y_train = y[:N]
        self.X_test = X[N:]
        self.y_test = y[N:]
           
    def train_LR(self):             
        self.LR = LogisticRegression()
        self.LR.fit(self.X_train,self.y_train)
        
    def train_SVC(self):
        self.SVC = SVC(probability=True)
        self.SVC.fit(self.X_train,self.y_train)
                
    def test(self,classifier):
        print classifier.score(self.X_test,self.y_test)
        
    def logloss(self,classifier):
        pred = classifier.predict_proba(self.X_test)
        s = 0
        for p in range(len(pred)):
            prob_one = pred[p][1]
            prob_zero = pred[p][0]
            truth = self.y_test[p]
            s += (truth*np.log(prob_one) + (1-truth)*np.log(prob_zero))
        print -s/len(self.y_test)
            
    #Predicts the probabilistic outcome of a single game________________________
    def predict_game(self,team1name,team2name,classifier):
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
        #game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
        game_feats = team1feats + team2feats
        pred = classifier.predict_proba(game_feats)
        return pred

if __name__ == '__main__':

    a = PredictNCAA()
    a.load_data()
    a.standardize_feats()
    a.training_testing_split()
    a.train_LR()
    a.train_SVC()
    a.test(a.SVC)
    a.test(a.LR)
    a.logloss(a.SVC)
    a.logloss(a.LR)
        
    team2 = 'Pittsburgh'
    team1 = 'Colorado'
    pred = a.predict_game(team1,team2,a.SVC)[0]
    print "Percent chances of winning:"
    print ("{team1}: %0.2f"%(pred[1]*100)).format(team1=team1)+"%"
    print ("{team2}: %0.2f"%(pred[0]*100)).format(team2=team2)+"%"


from __future__ import division
import pandas as pd
from lookup_kaggle_to_kenpom import team_lookup as kag_to_kp
from lookup_kenpom_to_kaggle import team_lookup as kp_to_kag
from lookup_kaggle_to_BlakeData import team_lookup as kag_to_bd
from lookup_BlakeData_to_kaggle import team_lookup as bd_to_kag
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.stats import nanmean,nanstd
from nltk import edit_distance
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

class PredictNCAA():
    
    def __init__(self):
        self.teams_kaggle = pd.read_csv('./teams.csv')
        self.feature_names = ['Rank','Pyth','AdjO','AdjOR','AdjD','AdjDR',
                            'AdjT','AdjTR','Luck','LuckR','PythSOS','PythSOSR',
                            'OppO','OppOR','OppD','OppDR','PythNCSOS','PythNCSOSR',
                            'Win %','SRS','SOS','Conf Win %','Home Win %','Away Win %',
                            'Points Scored','Points Against','FG','FGA','FG%','3P',
                            '3PA','3P%','FT','FTA','FT%','TRB','AST','STL','BLK',
                            'TOV','PF','FTr','3PAr','TS%','TRB%','AST%','BLK%',
                            'eFG%','TOV%','FT/FGA']
   
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
                    
    def load_blakeData(self):
        blakeDataAll = pd.read_csv('BlakeData.csv')
        blakeDataAll.Team = blakeDataAll.Team.apply(lambda x: re.sub(r'[0-9]*','',str(x)).strip().strip('.'))
        for year in range(2003,2015):
            blakeData = blakeDataAll[blakeDataAll['Year']==year]
            for id in self.teams_kaggle.id:
                if id in self.all_data[year].keys():
                    name = kag_to_bd[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()]
                    if isinstance(name,list): name = name[0]
                    if len(blakeData.Team[blakeData.Team==name]) == 0 and isinstance(kag_to_bd[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()],list): 
                        name = kag_to_bd[self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()][1]
                    if len(blakeData.Team[blakeData.Team==name]) == 0: 
                        name = self.teams_kaggle.name[self.teams_kaggle.id==int(id)].item()
                    if len(blakeData.Team[blakeData.Team==name]) != 0:
                        for column_name in blakeData.columns:
                            if column_name not in {"ORB","Year","Team"}:
                                self.all_data[year][id][column_name]=blakeData[column_name][blakeData['Year']==year][blakeData['Team']==name].item()
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
                    if teamid in self.all_data[year].keys():
                        if feat in self.all_data[year][teamid].keys():
                            all_vals.append(float(self.all_data[year][teamid][feat]))
            self.means[feat] = nanmean(all_vals)
            self.stdevs[feat] = nanstd(all_vals)
       
    #Used to find the teamname with smallest edit distance______________________ 
    def get_teamname(self,teamname):
        try:
            return kp_to_kag[teamname]
        except KeyError:
            try:
                return bd_to_kag[teamname]
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
                            game_feats = [team2feats[i] - team1feats[i] for i in range(len(team1feats))]
                            actual_label = 0
                        else:
                            game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
                            actual_label = team1wins
                    else: 
                        game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
                        actual_label = team1wins
                    if actual_label: num_ones += 1
                    else: num_zeros += 1
                            
                    X.append(game_feats)
                    y.append(actual_label)
        N = int(.7*len(X))
        self.X_full = X
        self.y_full = y
        self.X_train = X[:N]
        self.y_train = y[:N]
        self.X_test = X[N:]
        self.y_test = y[N:]
           
    def train_LR(self,X,y):             
        self.LR = LogisticRegression(fit_intercept = False)
        self.LR.fit(X,y)
        
    def train_SVC(self,X,y):
        self.SVC = SVC(probability=True,C=1e5,gamma=1e-5)
        self.SVC.fit(X,y)

    def train_RF(self,X,y):
        self.RF = RandomForestClassifier(n_estimators=10,max_features=6,min_samples_split=10)
        self.RF.fit(X,y)
                
    def test(self,classifier,X,y):
        print classifier.score(X,y)
        
    def test_ensemble(self,X,y):
        labels = a.RF.predict(X)
        pred_SVC = a.SVC.predict_proba(X)
        pred_LR = a.LR.predict_proba(X)
        agree = []
        pred = []
        for i in range(len(labels)):
            if pred_LR[i][labels[i]] > pred_SVC[i][labels[i]]:
                pred.append(pred_LR[i][1])
                if (pred_LR[i][1] < .5 and y[i]) or (pred_LR[i][1] > .5 and not y[i]):
                    agree.append(0)
                else: agree.append(1)
            else:
                pred.append(pred_SVC[i][1])
                if (pred_SVC[i][1] < .5 and y[i]) or (pred_SVC[i][1] > .5 and not y[i]):
                    agree.append(0)
                else: agree.append(1)
        print np.mean(agree) 
        
        s = 0
        for p in range(len(pred)):
            prob_one = pred[p]
            truth = y[p]
            s += (truth*np.log(prob_one) + (1-truth)*np.log(1-prob_one))
        print -s/len(y)
                          
          
    def logloss(self,classifier,X,y):
        pred = classifier.predict_proba(X)
        s = 0
        for p in range(len(pred)):
            prob_one = pred[p][1]
            prob_zero = pred[p][0]
            truth = y[p]
            s += (truth*np.log(prob_one) + (1-truth)*np.log(prob_zero))
        print -s/len(y)
            
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
        game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
        pred = classifier.predict_proba(game_feats)
        return pred
        
    def predict_ensemble(self,team1name,team2name):
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
        game_feats1 = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
        pred1 = self.RF.predict(game_feats1)[0]
        game_feats2 = [team2feats[i] - team1feats[i] for i in range(len(team1feats))]
        pred2 = self.RF.predict(game_feats2)[0]
        svcpred = self.SVC.predict_proba(game_feats1)
        lrpred = self.LR.predict_proba(game_feats1)
        if pred1 == pred2:
            return lrpred
        else:
            if lrpred[0][pred1] > svcpred[0][pred1]:
                return lrpred
            else:
                return svcpred
        
    def process_tournament(self,gameid,classifier):
        team1id = int(gameid.split('_')[1])
        team2id = int(gameid.split('_')[-1])
        year=2014
        team1feats = [(float(self.all_data[year][team1id][x]) - self.means[x])/self.stdevs[x] 
                    for x in self.feature_names]
        team2feats = [(float(self.all_data[year][team2id][x]) - self.means[x])/self.stdevs[x]
                            for x in self.feature_names]
        game_feats = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
        pred = classifier.predict_proba(game_feats)
        return pred[0][1]
        
    def process_tournament_ensemble(self,gameid):
        team1id = int(gameid.split('_')[1])
        team2id = int(gameid.split('_')[-1])
        year=2014
        team1feats = [(float(self.all_data[year][team1id][x]) - self.means[x])/self.stdevs[x] 
                    for x in self.feature_names]
        team2feats = [(float(self.all_data[year][team2id][x]) - self.means[x])/self.stdevs[x]
                            for x in self.feature_names]
        game_feats1 = [team1feats[i] - team2feats[i] for i in range(len(team1feats))]
        pred1 = self.RF.predict(game_feats1)[0]
        svcpred = self.SVC.predict_proba(game_feats1)
        lrpred = self.LR.predict_proba(game_feats1)
        game_feats2 = [team2feats[i] - team1feats[i] for i in range(len(team1feats))]
        pred2 = self.RF.predict(game_feats2)[0]
        if pred1 == pred2:
            return lrpred[0][1]
        else:
            if lrpred[0][pred1] > svcpred[0][pred1]:
                return lrpred[0][1]
            else:
                return svcpred[0][1]
        
    def tune_SVC():
        C_range = 10.0 ** np.arange(-2, 9)
        gamma_range = 10.0 ** np.arange(-5, 4)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedKFold(y=a.y_full, n_folds=3)
        grid = GridSearchCV(a.SVC, param_grid=param_grid, cv=cv)
        grid.fit(a.X_full, a.y_full)
        print("The best classifier is: ", grid.best_estimator_)

    def tune_RF():
        min_samples = np.arange(1,10)
        max_features = np.arange(3,15)
        param_grid = dict(min_samples_split=min_samples,
                            max_features=max_features)
        cv = StratifiedKFold(y=a.y_full, n_folds=10)
        grid = GridSearchCV(a.RF, param_grid=param_grid, cv=cv)
        grid.fit(a.X_full, a.y_full)
        print("The best classifier is: ", grid.best_estimator_)

if __name__ == '__main__':

    a = PredictNCAA()
    a.load_data()
    a.load_blakeData()
    a.standardize_feats()
    a.training_testing_split()
    a.train_LR(a.X_full,a.y_full)
    a.train_SVC(a.X_full,a.y_full)
    a.train_RF(a.X_full,a.y_full)
    
    print "SVC Test"
    a.test(a.SVC,a.X_test,a.y_test)
    a.logloss(a.SVC,a.X_test,a.y_test)

    print "LR Test"
    a.test(a.LR,a.X_test,a.y_test)
    a.logloss(a.LR,a.X_test,a.y_test)

    print "RF Test"
    a.test(a.RF,a.X_test,a.y_test)
    a.logloss(a.RF,a.X_test,a.y_test)

    print "Ensemble Test"
    a.test_ensemble(a.X_test,a.y_test)
        
    team1 = 'Albany'
    team2 = 'Mt St Mary\'s'
    #pred = a.predict_ensemble(team1,team2)[0]
    pred = a.predict_game(team1,team2,a.SVC)[0]
    print "Percent chances of winning:"
    print ("{team1}: %0.2f"%(pred[1]*100)).format(team1=team1)+"%"
    print ("{team2}: %0.2f"%(pred[0]*100)).format(team2=team2)+"%"

    #submit = pd.read_csv('sample_submission.csv')
    #submit.pred = submit.id.apply(lambda x: a.process_tournament(x,a.LR))
    #submit.to_csv('submission_LR.csv',index=False)
    #submit.pred = submit.id.apply(lambda x: a.process_tournament(x,a.SVC))
    #submit.to_csv('submission_SVC.csv',index=False)
    #submit.pred = submit.id.apply(lambda x: a.process_tournament_ensemble(x))
    #submit.to_csv('submission_ensemble.csv',index=False)

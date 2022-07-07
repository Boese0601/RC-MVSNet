clc
clear

resultPath1='/disk1/dan/di2/cascade_train_history/baseline_PMVS/collected_points_eval_0.75/TotalStat_binary.mat';
x1=load(resultPath1);
resultPath2='/disk1/dan/di2/cascade_train_history/baseline_PMVS/collected_points_eval_0.5/TotalStat_binary.mat';
x2=load(resultPath2);
resultPath3='/disk1/dan/di2/cascade_train_history/baseline_PMVS/collected_points_eval_0.25/TotalStat_binary.mat';
x3=load(resultPath3);
data1=x1.BaseStat;
data2=x2.BaseStat;
data3=x3.BaseStat;

% Stl=[data2.MeanStl;data3.MeanStl];
% Data=[data2.MeanData;data3.MeanData];

Stl=[data1.MeanStl;data2.MeanStl;data3.MeanStl];
Data=[data1.MeanData;data2.MeanData;data3.MeanData];
meanall=(Stl+Data)/2;
[min_value,index]=min(meanall,[],1)

overall= mean(min_value)
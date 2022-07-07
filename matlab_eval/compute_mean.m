resultPath='/home/dichang/code/related_work/u-mvsnet/dtu_eval/TotalStat_mvsnet.mat';
x=load(resultPath);
data=x.BaseStat;
comp=mean(data.MeanStl)
acc=mean(data.MeanData)
overall= (comp+acc)/2
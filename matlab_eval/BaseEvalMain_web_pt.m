clear all
close all
format compact
clc

% script to calculate distances have been measured for all included scans (UsedSets)


gt_dataPath = '/home/dichang/datasets/SampleSet/MVS_Data'

[dataPath,resultsPath]=getPaths();

dataPaths = {'/home/dichang/code/related_work/u-mvsnet'}


resultsPaths = {'/home/dichang/code/related_work/u-mvsnet/dtu_eval'}

light_string='l3'; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
method_string = 'mvsnet'
% get sets used in evaluation
UsedSets=GetUsedSets;

dst=0.2;    %Min dist between points when reducing

for i=1:length(dataPaths)
    dataPath = dataPaths{i};
    resultsPath = resultsPaths{i};
    parfor (j=1:length(UsedSets),10)
    % for j=1:length(UsedSets)
    % for cSet=UsedSets % for-loop is made to run through all sets used in the evaluation, set to 1 or 6 for the sample set
        cSet=UsedSets(j)
        %input data name
        DataInName=[dataPath sprintf('/%s%03d_%s.ply',lower(method_string),cSet,light_string)]
    
        %% 
        
        %results name
        EvalName=[resultsPath sprintf('/%s_%03d_%s.mat',lower(method_string),cSet,light_string)]
        
        disp(DataInName)
        disp(EvalName)
        
        %check if file is already computed
        if(~exist(EvalName,'file'))
            
            time=clock;time(4:5), drawnow
            
            tic
            Mesh = plyread(DataInName);
            Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
            toc
            
            BaseEval=PointCompareMain(cSet,Qdata,dst,gt_dataPath);
            
            disp('Saving results'), drawnow
            toc
            parsave(EvalName,BaseEval);
            toc
            
            % write obj-file of evaluation
            BaseEval2Obj_web(BaseEval,method_string, resultsPath)
            toc
            time=clock;time(4:5), drawnow
        end
    end
end

disp('All Done!')







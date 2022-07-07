function output = BaseEvalMain_web_pt_fn(UsedSets, dataPaths, resultsPaths)
%BaseEvalMain_web_pt_fn - Description
%
% Syntax: output = BaseEvalMain_web_pt_fn(input)
%
% Long description

    % clear all
    % close all
    format compact
    clc

    % script to calculate distances have been measured for all included scans (UsedSets)

    gt_dataPath = '/ssddata/datasets/UCSNet_data/SampleSet/MVS_Data'

    % [dataPath,resultsPath]=getPaths();

    % dataPaths = {'/ssddata/zmiaa/mvs-net/binary-mvs-exp/testing_20210222_1/collected_points',
    % '/ssddata/zmiaa/mvs-net/binary-mvs-exp/testing_20210222_1/collected_points_prob_0.6'}
    % resultsPaths = {'/ssddata/zmiaa/mvs-net/binary-mvs-exp/testing_20210222_1/collected_points_eval',
    % '/ssddata/zmiaa/mvs-net/binary-mvs-exp/testing_20210222_1/collected_points_prob_0.6_eval'}

    light_string='l3'; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
    method_string = 'binary'
    % get sets used in evaluation
    % UsedSets=GetUsedSets;

    dst=0.2;    %Min dist between points when reducing
    for i=1:length(dataPaths)
        dataPath = dataPaths{i};
        resultsPath = resultsPaths{i};
        % parfor (j=1:length(UsedSets), 10);
        for j=1:length(UsedSets)
        % for cSet=UsedSets % for-loop is made to run through all sets used in the evaluation, set to 1 or 6 for the sample set
            cSet=UsedSets(j)
            %input data name
            DataInName=[dataPath sprintf('/%s_%03d_%s.ply',lower(method_string),cSet,light_string)]
            
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
                save(EvalName,'BaseEval');
                toc
                
                % write obj-file of evaluation
                BaseEval2Obj_web(BaseEval,method_string, [resultsPath '/'])
                toc
                time=clock;time(4:5), drawnow
            end
        end
    end

    disp('All Done!')
    output = 0;
end









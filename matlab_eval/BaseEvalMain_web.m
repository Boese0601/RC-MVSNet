clear all
close all
format compact
clc

% script to calculate distances have been measured for all included scans (UsedSets)

addpath('MeshSupSamp_web\x64\Release');

[dataPath,resultsPath]=getPaths();

method_string='Tola'; %'Camp' %'Furu';
light_string='l7'; %'l7'; l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_IJCV_'; %results naming
        settings_string='';
    case 'Surfaces'
        eval_string='_SurfEval_Trim_IJCV_'; %results naming
        settings_string='_surf_11_trim_8'; %poisson settings for surface input
end

% get sets used in evaluation
if(strcmp(light_string,'l7'))
    UsedSets=GetUsedLightSets;
    eval_string=[eval_string 'l7_'];
else
    UsedSets=GetUsedSets;
end

dst=0.2;    %Min dist between points when reducing

for cSet=UsedSets % for-loop is made to run through all sets used in the evaluation, set to 1 or 6 for the sample set
    
    %input data name
    DataInName=[dataPath sprintf('/%s/%s/%s%03d_%s%s.ply',representation_string,lower(method_string),lower(method_string),cSet,light_string,settings_string)]
    
    %results name
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']
    
    %check if file is already computed
    if(~exist(EvalName,'file'))
        
        time=clock;time(4:5), drawnow
        
        tic
        Mesh = plyread(DataInName);
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        
        if(strcmp(representation_string,'Surfaces'))
            %upsample triangles
            Tri=cell2mat(Mesh.face.vertex_indices)';
            Qdata=MeshSupSamp(Qdata,Tri,dst);
        end
        toc
        
        BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);
        
        disp('Saving results'), drawnow
        toc
        save(EvalName,'BaseEval');
        toc
        
        % write obj-file of evaluation
        BaseEval2Obj_web(BaseEval,method_string, resultsPath)
        toc
        time=clock;time(4:5), drawnow
    end
end

% celebrate with a fanfare
load handel
sound(y,Fs)






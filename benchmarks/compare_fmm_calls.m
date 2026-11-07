function report = compare_fmm_calls(input)
%COMPARE_FMM_CALLS Compare two implementations of the same half-space flow.
%
% TWO-CALL (default):
%   1. Stokes FMM on the physical sources AND their reflected images.
%   2. Scalar Laplace FMM on the IMAGES ONLY, for the wall correction.
%   MATLAB combines the Stokes field with this charge/dipole correction.
%
% ONE-CALL (optional):
%   Fortran folds the wall correction into four Laplace potentials and calls
%   the Laplace FMM once on physical sources AND images. Only the images
%   carry nonzero dipoles, but the general kernels process all dipole entries.
%   Fortran reconstructs the complete velocity and pressure from the result.
%
% Both methods recenter coordinates and use potentials and gradients only
% for velocity/pressure. Neither needs Laplace Hessians for these outputs.
% Saving a tree need not save time: one call adds work on zero dipole entries.
%
% The comparison checks fields against an independent direct Blake reference,
% computed wall velocities, reciprocity, then velocity timings as sizes grow.
% All reported times use MATLAB TIC/TOC, outside the native call. Accuracy
% checks are outside the timed sections; no test requires a speedup.
%
% Usage, after setpath and blake.build_onecall:
%   report = compare_fmm_calls;
%   report = compare_fmm_calls(struct('sizes',[1024 4096], ...
%       'ratios',[0.25 1 4],'repetitions',3,'save',true));
%
% Options: sizes = source counts; ratios = target/source count ratios;
% tol = FMM tolerance; repetitions = timed pairs per case and scope;
% seed = random seed; save = write the report and CSV tables under results/.
% Defaults below run a full scaling comparison and can take several minutes.

%% Configuration and dependency record
if nargin == 0
    input = struct();
end
config = struct('sizes',[1024 4096 16384 65536],'ratios',[0.25 1 4], ...
    'tol',1e-8,'repetitions',5,'seed',481516,'save',false);
optionNames = fieldnames(input);
for optionIndex = 1:numel(optionNames)
    name = optionNames{optionIndex};
    assert(isfield(config,name),'Unknown comparison option: %s',name);
    config.(name) = input.(name);
end
validateattributes(config.sizes,{'double'},{'positive','integer','vector','finite'});
validateattributes(config.ratios,{'double'},{'positive','vector','finite'});
validateattributes(config.tol,{'double'},{'scalar','>',0,'<',1,'finite'});
validateattributes(config.repetitions,{'double'},{'positive','integer','scalar','finite'});
validateattributes(config.save,{'logical'},{'scalar'});

metadata = blake.internal.fmm.manifest();
assert(~isempty(metadata.onecall_mex_path),'Build the native backend with blake.build_onecall.');
report = struct('configuration',config,'metadata',metadata);
backendNames = {'two-call','one-call'};

fprintf(['\nComparing the same half-space representation:\n' ...
    '  Two calls: Stokes on physical sources + images, then Laplace on images only.\n' ...
    '             MATLAB assembles the complete field. This remains the default.\n' ...
    '  One call:  Fortran folds the wall correction into four Laplace potentials\n' ...
    '             on physical sources + images, then reconstructs the complete field.\n' ...
    '  Both recenter coordinates. Velocity/pressure need no Laplace Hessians.\n' ...
    '  All timings are measured in MATLAB; the native boundary includes packing/reconstruction.\n']);

%% 1. Field agreement and computed no-slip wall residuals
report.accuracy = compare_fields(config,backendNames);
fprintf('\nVelocity, pressure and wall residuals against direct Blake evaluation:\n');
disp(report.accuracy)

%% 2. Reciprocity after exchanging sources and targets
report.reciprocity = compare_reciprocity(config,backendNames);
fprintf('\nReciprocity defects (smaller is better):\n');
disp(report.reciprocity)

%% 3. Velocity speed as source and target counts increase
% Pressure is checked above, but is not requested in the timed evaluations.
[report.timings,report.samples] = compare_velocity_timings(config,backendNames);
fprintf('\nMedian velocity timings (speedup = two-call / one-call time):\n');
fprintf('Speedup above 1 favors one call; below 1 favors two calls.\n');
disp(report.timings)

%% 4. Optional saved report, including every timing repetition and build hashes
if config.save
    root = fileparts(fileparts(mfilename('fullpath')));
    destination = fullfile(root,'results');
    if ~isfolder(destination)
        mkdir(destination);
    end
    report.output_stem = [tempname(destination) '_fmm_calls'];
    save([report.output_stem '.mat'],'report');
    writetable(report.accuracy,[report.output_stem '_accuracy.csv']);
    writetable(report.reciprocity,[report.output_stem '_reciprocity.csv']);
    writetable(report.timings,[report.output_stem '_timings.csv']);
    writetable(report.samples,[report.output_stem '_samples.csv']);
end
end

function accuracy = compare_fields(config,backendNames)
% Compare velocity/pressure and raw wall flow with an independent direct reference.

% Four geometries exercise cancellation, physical units and recentering.
% Quantized coordinates preserve their separations after the large translation.
fixtureNames = {'bulk and wall','shallow sources', ...
    'nonzero wall and viscosity','large horizontal translation'};
sourceHeights = [1,1e-3,0.25,0.5];
wallHeights = [0,0,1.5,1.5];
viscosities = [1,1,2.5,2.5];
sourceCount = 1200;
targetCount = 37;
wallCount = 7;
errorLimit = 100*config.tol + 1e-12;
rows = cell(2*numel(fixtureNames),1);
rowIndex = 0;

for fixtureIndex = 1:numel(fixtureNames)
    rng(config.seed + fixtureIndex);
    height = sourceHeights(fixtureIndex);
    sources = round(4096*rand(3,sourceCount))/4096;
    sources(3,:) = height*(0.5 + sources(3,:));
    targets = round(4096*rand(3,targetCount))/4096;
    targets(3,:) = height*(0.2 + targets(3,:));
    targets(3,1:wallCount) = 0;

    % A target above the sources forces a nonzero vertical FMM center.
    % The folded potentials must still use the original height above the wall.
    targets(3,end) = 4;
    offset = [0;0;wallHeights(fixtureIndex)];
    if fixtureIndex == 4
        offset(1:2) = [2^30;-2^30];
    end
    sources = sources + offset;
    targets = targets + offset;
    forces = randn(3,sourceCount);

    % The direct Blake tensor is independent of the image representation.
    % Diagnostics provide a scale before cancellation of large image terms.
    options = struct('wall_height',wallHeights(fixtureIndex), ...
        'mu',viscosities(fixtureIndex),'tol',config.tol, ...
        'requested',{{'pressure'}},'diagnostics',true,'method','blake');
    [reference,referenceInfo] = blake.fields(sources,forces,targets,options);
    constituentScale = referenceInfo.constituent_scale;

    options.method = 'fmm';
    options.diagnostics = false;
    fields = cell(1,2);
    backendInfo = cell(1,2);
    for backendIndex = 1:2
        options.fmm_backend = backendNames{backendIndex};
        [fields{backendIndex},backendInfo{backendIndex}] = ...
            blake.fields(sources,forces,targets,options);
    end

    for backendIndex = 1:2
        computed = fields{backendIndex};
        info = backendInfo{backendIndex};
        velocityError = blake.metrics(computed.velocity,reference.velocity,constituentScale);
        backendDifference = blake.metrics(computed.velocity,fields{1}.velocity,constituentScale);
        pressureError = norm(computed.pressure-reference.pressure)/norm(reference.pressure);
        pressureDifference = norm(computed.pressure-fields{1}.pressure)/norm(reference.pressure);

        % Expected wall velocity is zero, so relative error against it is undefined.
        % Report both raw speed and speed divided by the constituent scale.
        % Never overwrite the computed wall values to force no-slip.
        wallSpeed = vecnorm(computed.velocity(:,1:wallCount));
        wallAbsolute = max(wallSpeed);
        wallNormalized = max(wallSpeed./constituentScale(1:wallCount));

        rowIndex = rowIndex + 1;
        rows{rowIndex} = struct('fixture',fixtureNames{fixtureIndex}, ...
            'backend',backendNames{backendIndex}, ...
            'velocity_relative',velocityError.global_relative, ...
            'velocity_normalized',velocityError.constituent_normalized, ...
            'pressure_relative',pressureError, ...
            'difference_normalized',backendDifference.constituent_normalized, ...
            'wall_absolute',wallAbsolute,'wall_normalized',wallNormalized, ...
            'fmm_calls',info.fmm_call_count);

        % Accept numerical accuracy, never a timing improvement.
        assert(velocityError.constituent_normalized < errorLimit,'Velocity accuracy failed.');
        assert(wallNormalized < errorLimit,'No-slip wall check failed.');
        assert(pressureError < errorLimit,'Pressure accuracy failed.');
        assert(pressureDifference < errorLimit,'Pressure implementations disagree.');
        assert(backendDifference.constituent_normalized < errorLimit,'Implementations disagree.');
        if backendIndex == 2
            % These are the actual call count and derivative flags from the native guard.
            assert(info.fmm_call_count == 1 && ...
                info.laplace_source_flag == 0 && info.laplace_target_flag == 2);
        end
    end
end
accuracy = struct2table(vertcat(rows{:}));
end

function reciprocity = compare_reciprocity(config,backendNames)
% Check force-work symmetry for two clouds and every entry of a 3-by-3 pair tensor.

% The Lorentz identity is sum(f_B .* u_A_at_B) = sum(f_A .* u_B_at_A).
% The clouds are large enough to exercise multipole work, not just direct leaves.
rng(config.seed + 100);
pointsA = rand(3,4096);
pointsA(3,:) = pointsA(3,:) + 0.1;
pointsB = rand(3,2048);
pointsB(3,:) = pointsB(3,:) + 0.2;
forcesA = randn(size(pointsA));
forcesB = randn(size(pointsB));
rows = cell(2,1);

for backendIndex = 1:2
    options = struct('tol',config.tol,'fmm_backend',backendNames{backendIndex});
    velocityAtB = blake.velocity(pointsA,forcesA,pointsB,options);
    velocityAtA = blake.velocity(pointsB,forcesB,pointsA,options);
    workAB = sum(forcesB(:).*velocityAtB(:));
    workBA = sum(forcesA(:).*velocityAtA(:));

    % A norm-product scale avoids division by a small signed work sum.
    workScale = norm(forcesB(:))*norm(velocityAtB(:)) ...
        + norm(forcesA(:))*norm(velocityAtA(:));
    distributedDefect = abs(workAB-workBA)/workScale;

    % At one pair of points, G(A,B) should equal transpose(G(B,A)).
    % Unit forces in the three coordinate directions construct all nine entries.
    pointA = [0.2;0.1;0.8];
    pointB = [0.7;0.3;1.4];
    tensorAB = zeros(3);
    tensorBA = zeros(3);
    for direction = 1:3
        force = zeros(3,1);
        force(direction) = 1;
        tensorAB(:,direction) = blake.velocity(pointA,force,pointB,options);
        tensorBA(:,direction) = blake.velocity(pointB,force,pointA,options);
    end
    pairDefect = norm(tensorAB-tensorBA.','fro')/norm(tensorAB,'fro');

    rows{backendIndex} = struct('backend',backendNames{backendIndex}, ...
        'sources_A',size(pointsA,2),'sources_B',size(pointsB,2), ...
        'distributed_defect',distributedDefect,'pair_defect',pairDefect);
    errorLimit = 100*config.tol + 1e-12;
    assert(distributedDefect < errorLimit && pairDefect < errorLimit,'Reciprocity failed.');
end
reciprocity = struct2table(vertcat(rows{:}));
end

function [timings,samples] = compare_velocity_timings(config,backendNames)
% Measure complete/prepared velocity calls in MATLAB and retain every repetition.

% Complete: includes geometry preparation and its checks on every call.
% Prepared: reuses checked geometry through blake.apply, but still rebuilds
% the FMM tree. Forces and coordinates are identical for both backends.
scopes = {'complete','prepared'};
sampleRows = {};
summaryRows = {};
caseIndex = 0;
fprintf('\nVelocity timings only: complete calls and prepared-geometry calls.\n');

for sourceCount = reshape(config.sizes,1,[])
    for targetRatio = reshape(config.ratios,1,[])
        caseIndex = caseIndex + 1;
        rng(config.seed + 200 + caseIndex);
        targetCount = max(2,round(sourceCount*targetRatio));
        wallCount = max(1,round(targetCount/10));

        % Random unit-width clouds above the wall, with 10% wall targets.
        sources = rand(3,sourceCount) - 0.5;
        sources(3,:) = 0.1 + rand(1,sourceCount);
        targets = rand(3,targetCount) - 0.5;
        targets(3,:) = 0.1 + rand(1,targetCount);
        targets(3,1:wallCount) = 0;
        forces = randn(3,sourceCount);
        geometry = blake.prepare(sources,targets);
        lastVelocity = cell(1,2);

        for scopeIndex = 1:numel(scopes)
            scope = scopes{scopeIndex};
            totalSeconds = zeros(config.repetitions,2);
            backendSeconds = zeros(config.repetitions,2);

            % Warm each exact case before collecting samples (JIT, MEX loading, etc.).
            for backendIndex = 1:2
                options = struct('tol',config.tol,'fmm_backend',backendNames{backendIndex});
                if strcmp(scope,'complete')
                    blake.velocity(sources,forces,targets,options);
                else
                    blake.apply(geometry,forces,options);
                end
            end

            for repetition = 1:config.repetitions
                % Alternate order to reduce bias from one backend always running first.
                executionOrder = [1,2];
                if mod(repetition,2) == 0
                    executionOrder = [2,1];
                end
                for backendIndex = executionOrder
                    options = struct('tol',config.tol,'fmm_backend',backendNames{backendIndex});

                    % MATLAB times the whole public call. No native stage timers.
                    timer = tic;
                    if strcmp(scope,'complete')
                        [velocity,info] = blake.velocity(sources,forces,targets,options);
                    else
                        [velocity,info] = blake.apply(geometry,forces,options);
                    end
                    elapsed = toc(timer);

                    totalSeconds(repetition,backendIndex) = elapsed;
                    backendSeconds(repetition,backendIndex) = info.fmm_evaluation_seconds;
                    lastVelocity{backendIndex} = velocity;
                    sampleRows{end+1} = struct('ns',sourceCount,'nt',targetCount, ...
                        'scope',scope,'backend',backendNames{backendIndex}, ...
                        'repetition',repetition,'total_seconds',elapsed, ...
                        'fmm_seconds',info.fmm_evaluation_seconds, ...
                        'fmm_calls',info.fmm_call_count); %#ok<AGROW>
                end
            end

            % Report medians and the range of paired ratios, preserving variability.
            % Backend times also come from MATLAB; the one-call MEX includes its
            % packing/reconstruction, so complete/prepared totals are the fair comparison.
            medianTotal = median(totalSeconds,1);
            medianBackend = median(backendSeconds,1);
            pairedSpeedup = totalSeconds(:,1)./totalSeconds(:,2);
            speedup = medianTotal(1)/medianTotal(2);
            summaryRows{end+1} = struct('ns',sourceCount,'nt',targetCount, ...
                'scope',scope,'two_call_s',medianTotal(1),'one_call_s',medianTotal(2), ...
                'speedup',speedup,'paired_min',min(pairedSpeedup), ...
                'paired_max',max(pairedSpeedup), ...
                'two_call_fmm_s',medianBackend(1),'one_call_fmm_s',medianBackend(2)); %#ok<AGROW>
            fprintf('Ns=%d Nt=%d %s: two %.4gs, one %.4gs, speedup %.3gx\n', ...
                sourceCount,targetCount,scope,medianTotal(1),medianTotal(2),speedup);
        end

        % Check a subset of wall AND bulk targets, summing every source in the
        % independent direct reference. This potentially expensive check is untimed.
        wallTargets = round(linspace(1,wallCount,min(8,wallCount)));
        bulkTargets = wallCount + round(linspace(1,targetCount-wallCount,min(24,targetCount-wallCount)));
        selectedTargets = unique([wallTargets,bulkTargets]);
        [reference,referenceInfo] = blake.velocity(sources,forces,targets(:,selectedTargets), ...
            struct('method','blake','diagnostics',true));
        for backendIndex = 1:2
            velocityError = blake.metrics(lastVelocity{backendIndex}(:,selectedTargets), ...
                reference,referenceInfo.constituent_scale);
            assert(velocityError.constituent_normalized < 100*config.tol+1e-12, ...
                'Large-case direct check failed.');
        end
    end
end
timings = struct2table(vertcat(summaryRows{:}));
samples = struct2table(vertcat(sampleRows{:}));
end

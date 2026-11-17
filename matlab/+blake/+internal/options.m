function opts = options(input)
%OPTIONS Normalize public options once; reject spelling mistakes.
if nargin == 0, input = struct(); end
opts = struct('mu',1,'wall_height',0,'tol',1e-9,'method','fmm', ...
    'block_size',[256 512],'diagnostics',false,'requested',{{'velocity'}}, ...
    'normals',zeros(3,0),'stokes_backend','auto','fmm_backend','two-call', ...
    'correct_near_wall',false,'near_wall_distance',[]);
if ~isstruct(input) || ~isscalar(input)
    error('blake:InvalidOptions','Options must be a scalar structure.');
end
names = fieldnames(input);
for k = 1:numel(names)
    if ~isfield(opts,names{k})
        error('blake:InvalidOptions','Unknown option: %s.',names{k});
    end
    opts.(names{k}) = input.(names{k});
end
validateattributes(opts.mu,{'double'},{'real','finite','scalar','positive'});
validateattributes(opts.wall_height,{'double'},{'real','finite','scalar'});
validateattributes(opts.tol,{'double'},{'real','finite','scalar','>',0,'<',1});
validateattributes(opts.block_size,{'double'},{'real','finite','integer','positive','numel',2});
validateattributes(opts.diagnostics,{'logical'},{'scalar'});
validateattributes(opts.correct_near_wall,{'logical'},{'scalar'});
if ~isempty(opts.near_wall_distance)
    validateattributes(opts.near_wall_distance,{'double'}, ...
        {'real','finite','scalar','positive'});
elseif ~isequal(opts.near_wall_distance,[])
    error('blake:InvalidOptions','near_wall_distance must be [] or a positive scalar.');
end
opts.method = validatestring(opts.method,{'blake','ggv-direct','fmm'});
opts.stokes_backend=validatestring(opts.stokes_backend,{'auto','upstream','laplace-velocity'});
opts.fmm_backend=validatestring(opts.fmm_backend,{'two-call','one-call'});
if strcmp(opts.fmm_backend,'one-call') && strcmp(opts.stokes_backend,'laplace-velocity')
    error('blake:UnsupportedBackend','one-call uses its own folded Fortran composition.');
end
if ischar(opts.requested) || isstring(opts.requested)
    opts.requested = cellstr(opts.requested);
end
if ~iscellstr(opts.requested) || ~all(ismember(opts.requested, ...
        {'velocity','pressure','gradient','traction'}))
    error('blake:InvalidOptions','Invalid requested field.');
end
end

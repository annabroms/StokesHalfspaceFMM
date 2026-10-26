function opts = options(input)
%OPTIONS Normalize public options once; reject spelling mistakes.
if nargin == 0, input = struct(); end
opts = struct('mu',1,'wall_height',0,'tol',1e-9,'method','fmm', ...
    'block_size',[256 512],'diagnostics',false,'requested',{{'velocity'}}, ...
    'normals',zeros(3,0),'stokes_backend','auto');
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
opts.method = validatestring(opts.method,{'blake','ggv-direct','fmm'});
opts.stokes_backend=validatestring(opts.stokes_backend,{'auto','upstream','laplace-velocity'});
if ischar(opts.requested) || isstring(opts.requested)
    opts.requested = cellstr(opts.requested);
end
if ~iscellstr(opts.requested) || ~all(ismember(opts.requested, ...
        {'velocity','pressure','gradient','traction'}))
    error('blake:InvalidOptions','Invalid requested field.');
end
end

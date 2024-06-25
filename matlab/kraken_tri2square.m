function varargout = kraken_tri2square(Ctri, varargin)
% Csquare = kraken_tri2square(Ctri, varargin)
% 
% Convert the upper triangular portion of a square matrix back to symmetric
% square form. This is SPECIFICALLY DESIGNED for the way numpy/krakencoder
% extracts the upper triangular portion of the original matrix, which is
% NOT the same as Matlab would produce.
%
% If user provides 'tri_indices' (returned by this function or saved out of
%  the krakencoder python scripts), use those to directly assign edges back
%  to the square form.
% Otherwise, if user provides 'numroi' and 'k', we create the equivalent
%   tri_indices and fill the square matrix.
% If the user only provides 'k' (default=1), we compute the appropriate
%  numroi based on numedges and k, create tri_indices, and fill the matrix

% Inputs:
% Ctri: [1 x numedges] row vector
% 
% Optional inputs:
% tri_indices: [2 x numedges] matrix of zero-based element indices (from
%   krakencoder, or returned by this function or kraken_square2tri)
% numroi: (optional) number of rois, used for computing the square matrix 
%   size and for computing new tri_indices of those are not provided
% k: (default=1) how many rows off-diagonal. If k=0, the Ctri includes the
%   diagonal values, so fill those in as well. k=1, diagonal is excluded.
% diagval: (default=nan) Value to fill any entries not included in the
%   tri_indices (eg: the diagonal if k=1)
% return_indices: default=false. If true, return tri_indices as a second
%   output along with Csquare
%
% Outputs:
% C: [numroi x numroi] square matrix
% tri_indices: [2 x numedges] matrix of zero-based element indices (if 'return_indices' is true)
%
% Example:
% Cdata = ones(86,86);
% [Ctri, tri_indices] = kraken_square2tri(Cdata, 'k', 1, 'return_indices',true); 
% Cnew = kraken_tri2square(Ctri, 'tri_indices', tri_indices);

args = inputParser;
args.addParameter('tri_indices',[]);
args.addParameter('numroi',0);
args.addParameter('k',1);
args.addParameter('return_indices',false);
args.addParameter('diagval',nan);

args.parse(varargin{:});
args = args.Results;

tri_indices=args.tri_indices;
numroi=args.numroi;
k=args.k;
diagval=args.diagval;
return_indices=args.return_indices;

if(size(Ctri,1)==numel(Ctri))
    %if input was an [edges x 1] vector, transpose so we always have [subj x edges]
    Ctri=Ctri.';
end
numsubj=size(Ctri,1);
numedges=size(Ctri,2);

if(numsubj>1)
    error('Multiple subjects provided. Must one row at a time');
end

if(isempty(tri_indices) && numroi <= 0)
    %no tri_indices and no numroi
    %compute numroi from edges and k and generate new tri_indices
    numroi=compute_numroi_from_numedges_k(numedges,k);
    
    %numpy fills in an order that is equivalent to tril(-k)
    %which we can recreate with triu(k)^T
    [tri_col, tri_row]=find(triu(true(numroi,numroi),k)');

    %make zero-based tri_indices for possible return
    tri_indices=[reshape(tri_row,1,[]); reshape(tri_col,1,[])]-1;

elseif(isempty(tri_indices))
    %no tri_indices but numroi provided
    %numpy fills in an order that is equivalent to tril(-k)
    %which we can recreate with triu(k)^T
    [tri_col, tri_row]=find(triu(true(numroi,numroi),k)');

    %make zero-based tri_indices for possible return
    tri_indices=[reshape(tri_row,1,[]); reshape(tri_col,1,[])]-1;
else
    %tri_indices was provided
    tri_row=tri_indices(1,:)+1;
    tri_col=tri_indices(2,:)+1;
    if(numroi<=0)
        numroi=max([tri_row(:); tri_col(:)]);
    end
end

Csquare=diagval*ones(numroi,numroi);
Csquare(sub2ind([numroi,numroi], tri_row, tri_col))=Ctri;
Csquare(sub2ind([numroi,numroi], tri_col, tri_row))=Ctri;

if(return_indices)
    varargout = {Csquare, tri_indices};
else
    varargout = {Csquare};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function numroi = compute_numroi_from_numedges_k(numedges,k)
%given a number of edges and a diag offset, compute the original
%square matrix dimension
if(k<0)
    numroi = (sqrt(8*k*k - 8*k + 8*numedges + 1) +2*k - 1)/2;
else
    numroi=k+(sqrt(1+8*numedges)-1)/2;
end
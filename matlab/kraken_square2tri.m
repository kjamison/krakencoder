function varargout = kraken_square2tri(C, varargin)

args = inputParser;
args.addParameter('tri_indices',[]);
args.addParameter('k',1);
args.addParameter('return_indices',false);

args.parse(varargin{:});
args = args.Results;

tri_indices=args.tri_indices;
k=args.k;
return_indices=args.return_indices;

numsubj=1;
numroi=size(C,2);

if(isempty(tri_indices))
    %numpy fills in an order that is equivalent to tril(-k)
    %which we can recreate with triu(k)^T
    [tri_col, tri_row]=find(triu(true(numroi,numroi),k)');
    %make zero-based tri_indices for possible return
    tri_indices=[reshape(tri_row,1,[]); reshape(tri_col,1,[])]-1;
else
    %tri_indices was provided
    tri_row=tri_indices(1,:)+1;
    tri_col=tri_indices(2,:)+1;
end
numedges=numel(tri_row);

Ctri=reshape(C(sub2ind([numroi,numroi], tri_row, tri_col)),[],numedges);

if(return_indices)
    varargout={Ctri, tri_indices};
else
    varargout={Ctri};
end

function sz = networkInputSize(net)
%NETWORKINPUTSIZE Network input size
%   sz = networkInputSize(NET) returns the input size of the single input
%   network NET. NET must have an image, 3-D image, sequence, or feature
%   input layer.

if ~isscalar(net.InputNames)
    error("Networks with multiple inputs not supported.")
end

inputName = net.InputNames{1};
layer = getLayer(net,inputName);

supportedLayers = [ 
    "ImageInputLayer"
    "Image3DInputLayer"
    "SequenceInputLayer" 
    "FeatureInputLayer"];

if ~contains(class(layer),supportedLayers)
    error("Input layer must be an image, 3-D image, sequence, or feature input layer.")
end

sz = layer.InputSize;

end


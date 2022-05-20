classdef CustomBackpropReluLayer < nnet.layer.Layer
    % CustomBackpropReluLayer  Relu layer with customizable backward
    %
    % This custom layer is intended for use in gradient attribution
    % visualizations. It implements standard backprop, Zeiler-Fergus
    % backprop and guided backprop.
    %
    % Example:
    %   layer = CustomBackpropReluLayer;
    %   layer.BackpropMode = "zeiler-fergus";    
    
    %   Copyright 2019 The MathWorks, Inc.

    properties
        BackpropMode (1,1) string {mustBeMember(BackpropMode,["backprop", "zeiler-fergus", "guided-backprop"])} = "backprop"
    end
    
    methods        
        function Z = predict(~, X)
            % Forward pass is usual ReLu function
            
            Z = max(X, 0);
        end

        function dLdX = backward(layer, X, ~, dLdZ, ~)
            % Backward pass can be modified from the conventional ReLU
            % backward.
            
            switch layer.BackpropMode
                case "backprop"
                    dLdX = (X > 0) .* dLdZ;
                case "zeiler-fergus"
                    dLdX = (dLdZ > 0) .* dLdZ;
                case "guided-backprop"
                    dLdX = (X > 0) .* (dLdZ > 0) .* dLdZ;
            end

        end
    end
end
%% init

% read images
imgDB = im2gray(imread("data/imgDB1.jpg"));

imgW = 512;
imgH = 512;
imgMeas = im2gray(imread("data/imgMeas1.jpg"));
imgMeas = imcrop(imgMeas, [0, 0, imgW, imgH]);

% methods for feature detection
% 1: BRISK 2:FAST 3: Harris 4: KAZE 5: MinEigen
% 6: MSER  7:ORB  8: SIFT   9: SURF
detectFeature = cell(1,9);
detectFeature{1} = @detectHarrisFeatures;
detectFeature{2} = @detectFASTFeatures;
detectFeature{3} = @detectHarrisFeatures;
detectFeature{4} = @detectKAZEFeatures;
detectFeature{5} = @detectMinEigenFeatures;
detectFeature{6} = @detectMSERFeatures;
detectFeature{7} = @detectORBFeatures;
detectFeature{8} = @detectSIFTFeatures;
detectFeature{9} = @detectSURFFeatures;
detectFeatureName = {'BRISK', 'FAST', 'Harris', 'KAZE', ...
                    'MinEigen', 'MSER', 'ORB', 'SIFT','SURF'};

% methods for extract features(descriptors)
% 1: interest point 2: local binary pattern(LBP)
% 3: histogram of oriented gradients (HOG)

% SET parameters
showFig = true;
detectFeatureMethods = [1:3]; % SET the method for feature detection (see line 11)


%% matching!!
for i = detectFeatureMethods 
    tic;
    % find the coners
    points1 = detectFeature{i}(imgDB);
    points2 = detectFeature{i}(imgMeas);

    % extract the neighborhood features
    [features1,valid_points1] = extractFeatures(imgDB,points1);
    [features2,valid_points2] = extractFeatures(imgMeas,points2);

    % match the features
    indexPairs = matchFeatures(features1,features2);

    % retrieve the locations of the corresponding points for each images
    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);

    % exclude the outliers, estimate the transformation matrix
    % M-estimator SAmple COnsesus(MSAC) algorithm, which is a variant of RANSAC
    % [x y 1]' = tform.T' * [u v 1]'
    % scale error: norm(tform.T(1,1:2))
    % position error: tform.T(3,1:2)
    [tform,inlierIdx] = estimateGeometricTransform2D(matchedPoints1,matchedPoints2,'similarity');
    inlierPts1 = matchedPoints1(inlierIdx,:);
    inlierPts2 = matchedPoints2(inlierIdx,:);

    ctime = toc;
    % visuallize
    if(showFig)
        figure;
        showMatchedFeatures(imgDB,imwarp(imgMeas,tform),inlierPts1,inlierPts2, 'montage', Parent=axes);
        title([detectFeatureName{i}, 'feature detection']);
    end
    
    disp(detectFeatureName{i});
    disp(['  POS ERROR: ', num2str(tform.T(3,1:2))]);
    disp(['  SCALE ERROR: ', num2str(norm(tform.T(1,1:2)))]);
    disp(['  COMPUTATIONAL TIME: ', num2str(ctime)]);
end
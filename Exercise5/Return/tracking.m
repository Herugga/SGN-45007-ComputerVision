%% Setup
% Create the webcam object.
cam = webcam();

% Create the video player object.
sz = [640 480];  % frame size
videoPlayer = vision.VideoPlayer('Position', [400 400 sz(1)+50 sz(2)+50]);

% Cascade detector instance
face_cascade = vision.CascadeObjectDetector('haarcascade_frontalface_default.xml');

% Parameters
face_cascade.ScaleFactor = 1.2;  % image pyramid scale
face_cascade.MergeThreshold = 5;  

% Point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);


%% USE MATLAB DOCUMENTATION TO FIND OUT HOW CERTAIN FUNCTIONS WORK.
% A few tips:
%  You should start by implementing the detection part first. 
%  Try drawing the trackable points in the detection part without saving them 
%  to p0 so you're able to see if the point coordinates are correct.
%  When finding the good points in the tracking part, use isFound as an index.


% Loop parameters
runLoop = true;
p0 = [];

while runLoop

    % Get a single next frame.
    img = imresize(snapshot(cam), 0.5);
    
    % Mirror
    img = fliplr(img);
    
    % Grayscale copy
    img_gray = rgb2gray(img);
    
    if size(p0, 1) <= 3
        % Detection
        img = insertText(img, [0,0], 'Detection');
        
        % Detect faces. Detections are in form 
        % (x_upperleft, y_upperleft, width, height)
        faces = face_cascade.step(img_gray);
    
        % Take the first face and get trackable points.
        if ~isempty(faces)
            % Extract ROI (face) from the grayscale frame
            % You can also crop this ROI even more to make sure only 
            % the face area is considered in the tracking.
            
            %%-your-code-starts-here-%%
            roi_gray = img_gray(faces(2):faces(2)+faces(4),faces(1):faces(1)+faces(3));  % replace with your implementation
            %%-your-code-ends-here-%%
            
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(roi_gray);
            p0 = points.Location;  % xy coordinates
            
            % Convert from ROI to image coordinates

            %%-your-code-starts-here-%%
            p0 = [p0(:,1)+faces(1), p0(:,2) + faces(2)];  % replace with you implementation
            %%-your-code-ends-here-%%
            
            % Initialized point tracker
            if ~isempty(p0)
                release(pointTracker);
                initialize(pointTracker, p0, img_gray);  
            end
        end
    else
        % Tracking
        img = insertText(img, [0,0], 'Tracking'); 

        % Calculate optical flow using pointTracker
        [p1, isFound] = step(pointTracker, img_gray);
        
        % Select good points. Use isFound to select valid found points
	    % from p1.      
        %%-your-code-starts-here-%%
        i = find(~isFound);
        p1(i,:) = [];
        %%-your-code-ends-here-%%
        
        % Draw points using e.g. insertMarker
        %%-your-code-starts-here-%%
        img = insertMarker(img, p1);
        %%-your-code-ends-here-%%

        % Update p0 for next iteration (which points should be kept?)
        %%-your-code-starts-here-%%
        p0(i,:) = [];
        p0 = p1;
        %%-your-code-ends-here-%%
        
        % Update pointTracker if p0 is found
        if ~isempty(p0)
            setPoints(pointTracker, p0);
        end
        
    end
 
    % Display the annotated video frame using the video player object.
    step(videoPlayer, img);
    
    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
    
end

% Release camera
clear cam

